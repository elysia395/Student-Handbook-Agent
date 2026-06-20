from pathlib import Path
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-m3",
            cache_folder=str(CACHE_DIR)
        )
