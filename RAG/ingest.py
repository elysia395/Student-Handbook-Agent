import re
import os
import fitz
import cv2
import numpy as np
import pytesseract
import chromadb
from collections import Counter
from .Embedder import Embedder

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TESS_CONFIG = "--psm 4"
TESS_LANG = "chi_sim"
OCR_DPI = 300
HEADER_CROP = 0.10
FOOTER_CROP = 0.06

HEADING_PATTERNS = [
    (re.compile(r'^第[一二三四五六七八九十百千零\d]+篇'), 1, "篇"),
    (re.compile(r'^第[一二三四五六七八九十百千零\d]+章'), 2, "章"),
    (re.compile(r'^第[一二三四五六七八九十百零\d]+条'), 3, "条"),
    (re.compile(r'^[（(][一二三四五六七八九十\d]+[）)]'), 4, "子条款"),
    (re.compile(r'^\d+[\.\．]'), 5, "子子条款"),
    (re.compile(r'^[一二三四五六七八九十]+、'), 6, "列表项"),
]


def pdf_page_to_img(doc, page_num, dpi=OCR_DPI):
    page = doc[page_num]
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def preprocess_image(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def split_dual_column(img):
    h, w = img.shape
    mid = w // 2
    left = img[:, :mid]
    right = img[:, mid:]
    return left, right


def crop_header_footer(img):
    h, w = img.shape
    top = int(h * HEADER_CROP)
    bottom = int(h * (1 - FOOTER_CROP))
    return img[top:bottom, :]


def ocr_image(img):
    text = pytesseract.image_to_string(img, lang=TESS_LANG, config=TESS_CONFIG)
    return [l.strip() for l in text.split("\n") if l.strip()]


def detect_heading_level(text):
    raw = text.strip()
    compact = re.sub(r'\s+', '', raw)
    for pattern, level, desc in HEADING_PATTERNS:
        if pattern.match(compact):
            # 标题行不应太长（阈值: 80字以内）
            if len(compact) <= 80:
                return level, raw
    return None


def detect_heading_level_batch(lines):
    results = []
    for line in lines:
        result = detect_heading_level(line)
        results.append(result)
    return results


def is_header_footer_candidate(text, freq_counter, threshold=0.4):
    count = freq_counter.get(text, 0)
    return count > 0


def build_freq_counter(all_lines, total_pages):
    counter = Counter()
    for page_lines in all_lines:
        seen = set()
        for line in page_lines:
            compact = re.sub(r'\s+', '', line).strip()
            if len(compact) > 2 and len(compact) < 30:
                key = compact[:20]
                if key not in seen:
                    counter[key] += 1
                    seen.add(key)
    return counter


def structure_chunk_hierarchical(ordered_blocks):
    chunks = []
    current_chunk_lines = []
    current_levels = [None] * 7
    current_chunk_page_start = ordered_blocks[0]["page"] if ordered_blocks else 0

    for block in ordered_blocks:
        text = block["text"]
        heading_result = block.get("heading")
        page = block.get("page", 0)

        if heading_result:
            level, title = heading_result
            if current_chunk_lines:
                chunks.append({
                    "text": "\n".join(current_chunk_lines),
                    "chapter": current_levels[2],
                    "article": current_levels[3],
                    "page_start": current_chunk_page_start,
                    "page_end": page,
                })
                current_chunk_lines = []
            current_levels[level] = title
            for l in range(level + 1, 6):
                current_levels[l] = None
            current_chunk_page_start = page
            current_chunk_lines.append(text)
        else:
            current_chunk_lines.append(text)

    if current_chunk_lines:
        chunks.append({
            "text": "\n".join(current_chunk_lines),
            "chapter": current_levels[2],
            "article": current_levels[3],
            "page_start": current_chunk_page_start,
            "page_end": ordered_blocks[-1]["page"],
        })

    return chunks


def inject_context(chunks):
    result = []
    for c in chunks:
        prefix_parts = []
        if c.get("chapter"):
            prefix_parts.append(c["chapter"])
        if c.get("article"):
            prefix_parts.append(c["article"])
        prefix = f"[{' → '.join(prefix_parts)}] " if prefix_parts else ""
        result.append(prefix + c["text"])
    return result


def split_by_delimiters(text, delimiters):
    pattern = "|".join(re.escape(d) for d in delimiters)
    parts = re.split(f"({pattern})", text)
    merged = []
    for i in range(0, len(parts) - 1, 2):
        merged.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1]:
        merged.append(parts[-1])
    return merged


def merge_small_chunks(parts, chunk_size):
    merged = []
    current = ""
    for p in parts:
        if len(current) + len(p) < chunk_size:
            current += p
        else:
            if current:
                merged.append(current)
            current = p
    if current:
        merged.append(current)
    return merged


def recursive_symbol_split(text, chunk_size=CHUNK_SIZE):
    if len(text) <= chunk_size:
        return [text]
    parts = split_by_delimiters(text, ["。", "！", "？", "；"])
    if len(parts) > 1:
        merged = merge_small_chunks(parts, chunk_size)
        if all(len(c) <= chunk_size for c in merged):
            return merged
    parts = split_by_delimiters(text, ["，", "、", "："])
    if len(parts) > 1:
        merged = merge_small_chunks(parts, chunk_size)
        if all(len(c) <= chunk_size for c in merged):
            return merged
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def add_overlap(chunks, overlap=CHUNK_OVERLAP):
    result = []
    for i, c in enumerate(chunks):
        if i > 0:
            prev_tail = chunks[i - 1][-overlap:]
            c = prev_tail + c
        result.append(c)
    return result


class Ingestor:
    def __init__(self):
        self.embedder = Embedder()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="student_handbook",
            metadata={"hnsw:space": "cosine"}
        )

    def ingest(self, pdf_path):
        count = self.collection.count()
        if count > 0:
            print(f"知识库已有 {count} 条记录，跳过导入")
            return count

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"共 {total_pages} 页，开始 OCR 处理...")

        all_page_lines = []
        all_page_blocks = []

        for page_num in range(total_pages):
            print(f"  处理页 {page_num + 1}/{total_pages}...", end="")
            img = pdf_page_to_img(doc, page_num)
            img = crop_header_footer(img)
            img = preprocess_image(img)

            h, w = img.shape
            if w > h * 1.3:
                left, right = split_dual_column(img)
                lines_left = ocr_image(left)
                lines_right = ocr_image(right)
                page_lines = [("l", l) for l in lines_left] + [("r", l) for l in lines_right]
            else:
                page_lines_full = ocr_image(img)
                page_lines = [("", l) for l in page_lines_full]

            all_page_lines.append([l for _, l in page_lines])

            for col, line in page_lines:
                heading = detect_heading_level(line)
                all_page_blocks.append({
                    "text": line,
                    "heading": heading,
                    "page": page_num,
                    "col": col,
                })
            print(f" {len(page_lines)} 行")

        doc.close()

        print("\n检测并去重页眉页脚...")
        freq = build_freq_counter(all_page_lines, total_pages)
        threshold = total_pages * 0.4
        hf_candidates = {t for t, c in freq.items() if c >= threshold and len(t) < 30}

        filtered_blocks = [b for b in all_page_blocks if re.sub(r'\s+', '', b["text"])[:20] not in hf_candidates]
        print(f"  移除 {len(all_page_blocks) - len(filtered_blocks)} 个页眉页脚候选")

        print("层级结构分块...")
        structured_chunks = structure_chunk_hierarchical(filtered_blocks)
        print(f"  共 {len(structured_chunks)} 个结构块")

        print("注入上下文前缀...")
        context_chunks = inject_context(structured_chunks)

        print("递归符号分块...")
        all_chunks = []
        for c in context_chunks:
            all_chunks.extend(recursive_symbol_split(c, CHUNK_SIZE))

        all_chunks = add_overlap(all_chunks, CHUNK_OVERLAP)
        print(f"  共 {len(all_chunks)} 个文档片段")

        print("生成嵌入向量...")
        embeddings = self.embedder.model.encode(all_chunks, show_progress_bar=True)

        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        print(f"成功导入 {len(all_chunks)} 个文档片段到知识库")
        return len(all_chunks)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：uv run python -m RAG.ingest <pdf文件路径>")
        sys.exit(1)
    ingestor = Ingestor()
    ingestor.ingest(sys.argv[1])
