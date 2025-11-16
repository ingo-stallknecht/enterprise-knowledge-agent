# app/rag/index.py
import os
import json
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np


class DocIndex:
    """
    Simple FAISS index + docstore wrapper.

    - build(X, records): overwrite full index + docstore
    - add(new_vecs, new_records): append vectors + records, then persist
    - query_dense(qv, k): dense retrieval
    - query_sparse(query, k): placeholder for future sparse retrieval
    - query_hybrid(qv, query, k, alpha): currently the same as dense
    - size(): number of records

    This version is **backward compatible** with older docstore formats:
      * If store_json is a list → treated directly as the records list.
      * If it's a dict → tries `["records"]` or the first list-of-dicts value.
    """

    def __init__(
        self,
        faiss_path: str = "data/index/handbook.index",
        store_path: str = "data/index/docstore.json",
    ):
        self.faiss_path = faiss_path
        self.store_path = store_path
        self.index: Optional[faiss.Index] = None
        self.records: List[Dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load FAISS index and docstore from disk, tolerant to legacy formats."""
        # Load FAISS index
        if os.path.exists(self.faiss_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
            except Exception:
                # Corrupt or incompatible index → start empty
                self.index = None
        else:
            self.index = None

        # Load docstore (JSON)
        self.records = []
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                raw = None

            # Legacy format: list of records
            if isinstance(raw, list):
                self.records = raw

            # Newer / other format: dict that may include 'records'
            elif isinstance(raw, dict):
                if isinstance(raw.get("records"), list):
                    self.records = raw["records"]
                else:
                    # Fallback: look for any list-of-dicts field
                    for v in raw.values():
                        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                            self.records = v
                            break
            # else: leave records as []

    def _ensure_loaded(self) -> None:
        """Lazily load index and records."""
        if self.index is None or self.records is None or len(self.records) == 0:
            self._load()

    def _persist(self) -> None:
        """Persist FAISS index + docstore to disk."""
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.faiss_path)

        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        # We store just the records list (simple, version-agnostic)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, X: np.ndarray, records: List[Dict]) -> None:
        """
        Overwrite the full index and docstore.

        Parameters
        ----------
        X : np.ndarray
            Float32 matrix of shape (n_samples, dim). If empty → index is empty.
        records : list of dict
            One entry per vector.
        """
        # Determine dimension
        if X is not None and X.ndim == 2 and X.shape[0] > 0:
            dim = X.shape[1]
        else:
            # Default MiniLM dimension; won't matter until vectors are added
            dim = 384

        self.index = faiss.IndexFlatIP(dim)
        self.records = list(records) if records is not None else []

        if X is not None and X.ndim == 2 and X.shape[0] > 0:
            self.index.add(X.astype("float32"))

        self._persist()

    def add(self, new_vecs: np.ndarray, new_records: List[Dict]) -> None:
        """
        Append vectors + records to existing index.
        If index does not exist, behaves like `build`.
        """
        if new_vecs is None or new_vecs.ndim != 2 or new_vecs.shape[0] == 0:
            return
        if not new_records:
            return

        self._ensure_loaded()

        dim = int(new_vecs.shape[1])
        # Initialize index if missing
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            self.records = []

        # If index is empty, make sure dim matches
        if self.index.ntotal == 0 and self.index.d != dim:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(new_vecs.astype("float32"))
        self.records.extend(list(new_records))
        self._persist()

    def query_dense(self, qv: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Dense retrieval using FAISS.
        Returns a list of (record, score).
        """
        if qv is None or qv.ndim != 2 or qv.shape[0] == 0:
            return []

        self._ensure_loaded()
        if self.index is None or not self.records:
            return []

        k = max(1, int(k))
        D, I = self.index.search(qv.astype("float32"), k)
        out: List[Tuple[Dict, float]] = []
        for j, idx in enumerate(I[0]):
            if 0 <= idx < len(self.records):
                out.append((self.records[idx], float(D[0, j])))
        return out

    def query_sparse(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Placeholder for TF-IDF / BM25 retrieval.
        Currently returns an empty list to keep the API stable.
        """
        _ = query  # unused
        _ = k
        return []

    def query_hybrid(
        self,
        qv: np.ndarray,
        query: str,
        k: int = 5,
        alpha: float = 0.6,
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid retrieval placeholder.

        Currently this simply calls `query_dense`. Hybridization with sparse
        scores can be added later while keeping this interface.
        """
        _ = query
        _ = alpha
        return self.query_dense(qv, k=k)

    def size(self) -> int:
        """
        Number of records in the docstore (best-effort, even if not loaded yet).
        """
        # Try in-memory first
        if self.records:
            return len(self.records)

        # Fallback: read from disk if needed
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    return len(raw)
                if isinstance(raw, dict):
                    if isinstance(raw.get("records"), list):
                        return len(raw["records"])
                    for v in raw.values():
                        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                            return len(v)
            except Exception:
                return 0
        return 0
