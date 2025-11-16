# app/rag/index.py
import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np

# Try to import FAISS; fall back to pure NumPy if not available
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - cloud may not have faiss
    faiss = None
    _FAISS_AVAILABLE = False


class DocIndex:
    """
    Simple index + docstore wrapper with FAISS-optional backend.

    - If FAISS is available:
        * Uses IndexFlatIP for inner-product similarity
        * Persists index with faiss.write_index(...)
    - Otherwise:
        * Stores vectors in a .npy file and performs NumPy dot-product search

    Public API:
        build(X, records)
        add(new_vecs, new_records)
        query_dense(qv, k)
        query_sparse(query, k)         # placeholder
        query_hybrid(qv, query, k, alpha)
        size()
    """

    def __init__(
        self,
        faiss_path: str = "data/index/handbook.index",
        store_path: str = "data/index/docstore.json",
    ):
        self.faiss_path = faiss_path
        self.store_path = store_path

        # FAISS backend
        self.index = None

        # NumPy backend
        self._vecs: Optional[np.ndarray] = None

        # Shared docstore
        self.records: List[Dict] = []

        # Lazy load state
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

    def _numpy_vecs_path(self) -> str:
        # Use same base path but with .npy extension for vectors
        base = self.faiss_path
        if not base.endswith(".npy"):
            base = base + ".npy"
        return base

    def _load(self):
        """Load index + records from disk if available."""
        self._ensure_dir()

        # Load records
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []
        else:
            self.records = []

        # Load FAISS or NumPy vectors
        if _FAISS_AVAILABLE and os.path.exists(self.faiss_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
            except Exception:
                self.index = None
        else:
            self.index = None
            np_path = self._numpy_vecs_path()
            if os.path.exists(np_path):
                try:
                    self._vecs = np.load(np_path).astype("float32")
                except Exception:
                    self._vecs = None
            else:
                self._vecs = None

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()

    def _persist(self):
        self._ensure_dir()
        # Persist records
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

        # Persist vectors depending on backend
        if _FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, self.faiss_path)
        else:
            if self._vecs is not None:
                np.save(self._numpy_vecs_path(), self._vecs.astype("float32"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, X: np.ndarray, records: List[Dict]):
        """Overwrite full index and docstore."""
        X = X.astype("float32") if X is not None else np.zeros((0, 384), dtype="float32")
        self.records = list(records) if records is not None else []

        if _FAISS_AVAILABLE:
            dim = X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else 384
            self.index = faiss.IndexFlatIP(dim)
            if X.shape[0] > 0:
                self.index.add(X)
            self._vecs = None  # not used in FAISS mode
        else:
            # Pure NumPy backend
            self.index = None
            self._vecs = X

        self._persist()

    def add(self, new_vecs: np.ndarray, new_records: List[Dict]):
        """
        Append vectors + records to existing index.
        If index does not exist yet, behaves like build().
        """
        if new_vecs is None or new_vecs.shape[0] == 0:
            return
        if not new_records:
            return

        self._ensure_loaded()
        new_vecs = new_vecs.astype("float32")

        if _FAISS_AVAILABLE:
            # Initialize index if missing
            if self.index is None:
                dim = new_vecs.shape[1]
                self.index = faiss.IndexFlatIP(dim)
            # If index is empty but dimension mismatched, recreate
            if self.index.ntotal == 0 and self.index.d != new_vecs.shape[1]:
                self.index = faiss.IndexFlatIP(new_vecs.shape[1])
            self.index.add(new_vecs)
        else:
            # NumPy backend: append rows
            if self._vecs is None:
                self._vecs = new_vecs
            else:
                if self._vecs.shape[1] != new_vecs.shape[1]:
                    # Dimension mismatch: rebuild from scratch with new_vecs only
                    self._vecs = new_vecs
                    self.records = []
            # Extend vectors by vertical stacking
            self._vecs = (
                new_vecs if self._vecs is None else np.vstack([self._vecs, new_vecs])
            )

        self.records.extend(new_records)
        self._persist()

    def _query_dense_numpy(self, qv: np.ndarray, k: int) -> List[Tuple[Dict, float]]:
        """NumPy-based dense retrieval for when FAISS is unavailable."""
        if self._vecs is None or self._vecs.shape[0] == 0 or not self.records:
            return []
        # qv shape: (1, d)
        sims = (qv @ self._vecs.T).reshape(-1)
        k = min(k, sims.shape[0])
        idxs = np.argsort(-sims)[:k]
        out: List[Tuple[Dict, float]] = []
        for j in idxs:
            if 0 <= j < len(self.records):
                out.append((self.records[j], float(sims[j])))
        return out

    def query_dense(self, qv: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Dense retrieval using FAISS or NumPy fallback.
        Returns a list of (record, score).
        """
        if qv is None or qv.shape[0] == 0:
            return []
        self._ensure_loaded()
        if _FAISS_AVAILABLE and self.index is not None and len(self.records) > 0:
            D, I = self.index.search(qv.astype("float32"), k)
            out: List[Tuple[Dict, float]] = []
            for j, idx in enumerate(I[0]):
                if 0 <= idx < len(self.records):
                    out.append((self.records[idx], float(D[0, j])))
            return out
        # Fallback: NumPy dense search
        return self._query_dense_numpy(qv.astype("float32"), k)

    def query_sparse(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        # Placeholder for TF-IDF / BM25 mode
        return []

    def query_hybrid(
        self, qv: np.ndarray, query: str, k: int = 5, alpha: float = 0.6
    ) -> List[Tuple[Dict, float]]:
        # Currently just dense retrieval; alpha unused
        return self.query_dense(qv, k)

    def size(self) -> int:
        """Number of records in the docstore."""
        self._ensure_loaded()
        if self.records:
            return len(self.records)
        # Fallback: read store from disk
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                return len(recs)
            except Exception:
                return 0
        return 0
