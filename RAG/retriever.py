import chromadb
import torch
from sentence_transformers import CrossEncoder
from pathlib import Path
from .Embedder import Embedder

CACHE_DIR = Path(__file__).resolve().parent.parent / "models"

# ── 全局参数 ──
INITIAL_RECALL = 20
VECTOR_SIMILARITY_BASE = 0.35
VECTOR_SIMILARITY_FALLBACK = 0.50
RERANK_MIN_SCORE = 0.45
RERANK_FALLBACK_SCORE = 0.30
FORCED_MIN_RESULTS = 5
FALLBACK_MAX_RESULTS = 3
SINGLE_DROP_THRESHOLD = 0.1
CONSECUTIVE_DROPS_THRESHOLD = 2
DEDUP_SIMILARITY_THRESHOLD = 0.9


class Retriever:
    def __init__(self):
        self.embedder = Embedder()

        try:
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_folder=str(CACHE_DIR),
            )
        except Exception as e:
            print(f"[WARN] Reranker 加载失败: {e}，将跳过 rerank 阶段")
            self.reranker = None

        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="student_handbook",
            metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, top_k: int = None) -> list[str]:
        top_k = top_k or FORCED_MIN_RESULTS

        if self.collection.count() == 0:
            return []

        # ═══ 阶段 1：向量粗召 & 初筛 ═══
        query_embedding = self.embedder.model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=INITIAL_RECALL,
            include=["documents", "distances"]
        )
        documents = results["documents"][0] if results.get("documents") else []
        distances = results["distances"][0] if results.get("distances") else []

        if not documents:
            return []

        similarities = [1 - d for d in distances]

        qualified = [
            (doc, sim) for doc, sim in zip(documents, similarities)
            if sim >= VECTOR_SIMILARITY_BASE
        ]

        if not qualified:
            fallback = [
                (doc, sim) for doc, sim in zip(documents, similarities)
                if sim >= VECTOR_SIMILARITY_FALLBACK
            ]
            if not fallback:
                return []
            fallback_docs = [doc for doc, _ in fallback[:FALLBACK_MAX_RESULTS]]
            return self._rerank_and_filter(query, fallback_docs, is_fallback=True)

        qualified_docs = [doc for doc, _ in qualified]

        if self.reranker:
            return self._rerank_and_filter(query, qualified_docs, is_fallback=False)
        return qualified_docs[:top_k]

    def _rerank_and_filter(self, query: str, documents: list[str], is_fallback: bool) -> list[str]:
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        min_score = RERANK_FALLBACK_SCORE if is_fallback else RERANK_MIN_SCORE
        max_return = FALLBACK_MAX_RESULTS if is_fallback else None

        valid_docs = [(doc, score) for doc, score in scored_docs if score >= min_score]

        if not valid_docs:
            if is_fallback:
                return []
            extreme_fallback = [(doc, score) for doc, score in scored_docs if score >= RERANK_FALLBACK_SCORE]
            return [doc for doc, _ in extreme_fallback[:FALLBACK_MAX_RESULTS]]

        if max_return:
            return [doc for doc, _ in valid_docs[:max_return]]

        if len(valid_docs) <= FORCED_MIN_RESULTS:
            return [doc for doc, _ in valid_docs]

        # ═══ 阶段 5：断崖动态截取 ═══
        final_docs = [doc for doc, _ in valid_docs[:FORCED_MIN_RESULTS]]
        consecutive_drops = 0

        for i in range(FORCED_MIN_RESULTS, len(valid_docs)):
            prev_score = valid_docs[i - 1][1]
            curr_score = valid_docs[i][1]
            drop = prev_score - curr_score

            if drop > SINGLE_DROP_THRESHOLD:
                consecutive_drops += 1
                if consecutive_drops >= CONSECUTIVE_DROPS_THRESHOLD:
                    break
            else:
                consecutive_drops = 0

            final_docs.append(valid_docs[i][0])

        return self._dedup(final_docs)

    def _dedup(self, documents: list[str]) -> list[str]:
        if len(documents) <= 1:
            return documents

        embeddings = self.embedder.model.encode(documents)
        keep = [True] * len(documents)

        for i in range(len(documents)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(documents)):
                if not keep[j]:
                    continue
                sim = float(
                    torch.nn.functional.cosine_similarity(
                        torch.tensor(embeddings[i]).unsqueeze(0),
                        torch.tensor(embeddings[j]).unsqueeze(0)
                    )
                )
                if sim > DEDUP_SIMILARITY_THRESHOLD:
                    keep[j] = False

        return [doc for doc, k in zip(documents, keep) if k]
