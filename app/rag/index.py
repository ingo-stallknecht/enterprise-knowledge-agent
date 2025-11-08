# app/rag/index.py
import faiss, numpy as np, json, os, pathlib
from typing import List, Dict, Optional

class DocIndex:
    """
    Simple FAISS (IP) + JSON docstore with incremental add().
    Records are a list of dicts with at least: {"text": ..., "source": ...}
    """
    def __init__(self, faiss_path="data/index/handbook.index", store_path="data/index/docstore.json"):
        self.faiss_path, self.store_path = faiss_path, store_path
        self.index: Optional[faiss.IndexFlatIP] = None
        self.records: List[Dict] = []

    # --------------- core I/O ---------------
    def _empty_index(self, dim: int = 384):
        self.index = faiss.IndexFlatIP(dim)
        self.records = []

    def load(self):
        """Load existing index/docstore or initialize empty if missing/corrupt."""
        faiss_ok = pathlib.Path(self.faiss_path).exists()
        store_ok = pathlib.Path(self.store_path).exists()
        if not (faiss_ok and store_ok):
            self._empty_index()
            return

        try:
            self.index = faiss.read_index(self.faiss_path)
        except Exception:
            self._empty_index()
            return

        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            if not isinstance(self.records, list):
                self.records = []
        except Exception:
            self.records = []

        # safety: mismatch → reset
        try:
            if self.index.ntotal != len(self.records):
                dim = self.index.d if self.index is not None else 384
                self._empty_index(dim)
        except Exception:
            self._empty_index()

    def save(self):
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_path)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    # --------------- build / add ---------------
    def build(self, X: np.ndarray, records: List[Dict]):
        """
        Full rebuild from embeddings X and records.
        """
        dim = X.shape[1] if len(X.shape) == 2 and X.shape[0] > 0 else 384
        self.index = faiss.IndexFlatIP(dim)
        if X.shape[0] > 0:
            self.index.add(X.astype("float32"))
        self.records = list(records)
        self.save()

    def add(self, X: np.ndarray, records: List[Dict]):
        """
        Incrementally append vectors + records.
        """
        if self.index is None:
            self._empty_index(X.shape[1] if X.size else 384)
        if X.shape[0] > 0:
            self.index.add(X.astype("float32"))
            self.records.extend(records)
            self.save()

    # --------------- queries ---------------
    def _ensure_loaded(self):
        if self.index is None or self.records is None:
            self.load()

    def size(self) -> int:
        self._ensure_loaded()
        try:
            return int(self.index.ntotal)
        except Exception:
            return 0

    def query_dense(self, qv, k=5):
        self._ensure_loaded()
        if self.index is None or self.size() == 0 or len(self.records) == 0:
            return []
        D, I = self.index.search(qv.astype("float32"), k)
        out = []
        for j, i in enumerate(I[0]):
            if 0 <= i < len(self.records):
                out.append((self.records[i], float(D[0, j])))
        return out

    def query_sparse(self, query, k=5):
        # Placeholder for TF-IDF
        return []

    def query_hybrid(self, qv, query, k=5, alpha=0.6):
        # Simple hybrid = dense-only for now
        return self.query_dense(qv, k)
