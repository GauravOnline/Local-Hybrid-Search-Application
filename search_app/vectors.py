"""Local sentence embeddings and FAISS similarity search."""

import os
from pathlib import Path

import faiss
import numpy as np

from .config import DIMENSION, MODEL_DIR


class SentenceEncoder:
    """Load a previously downloaded model; no network requests during search."""

    dimension = DIMENSION

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return normalized float32 embeddings suitable for cosine similarity."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self._model is None:
            if not self.model_dir.is_dir():
                raise RuntimeError("Model not found. Run python -m scripts.download_model first.")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            import torch

            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self._model = SentenceTransformer(str(self.model_dir), device="cpu", local_files_only=True)
        return np.asarray(self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)


class VectorIndex:
    """FAISS index using stable SQLite chunk IDs instead of row positions."""

    def __init__(self, dimension: int, path: Path):
        self.dimension = dimension
        self.path = path
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def rebuild(self, rows: list) -> None:
        """Reconstruct the derived index from committed chunk embeddings."""
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        if rows:
            ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
            vectors = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
            index.add_with_ids(np.ascontiguousarray(vectors), ids)
        temporary = self.path.with_suffix(".tmp")
        faiss.write_index(index, str(temporary))
        temporary.replace(self.path)
        self.index = index

    def search(self, vector: np.ndarray) -> list[tuple[int, float]]:
        """Return ranked chunk IDs and cosine scores."""
        if not self.index.ntotal:
            return []
        faiss.omp_set_num_threads(min(4, os.cpu_count() or 1))
        scores, ids = self.index.search(np.ascontiguousarray(vector.reshape(1, -1)), self.index.ntotal)
        return [(int(chunk_id), float(score)) for chunk_id, score in zip(ids[0], scores[0]) if chunk_id >= 0]
