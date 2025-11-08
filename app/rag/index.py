# app/rag/index.py
# Incremental FAISS index with ID mapping and targeted delete-by-source.
# - Uses IndexIDMap2 over IndexFlatIP so we can remove by vector IDs.
# - Persists a compact docstore mapping {id: {text, source}}.
# - Exposes build(), add(), remove_sources(), size(), load(), save(), query_*().

import os
import json
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss-cpu is required for indexing. pip install faiss-cpu") from e


class DocIndex:
    def __init__(self, faiss_path: str = "data/index/handbook.index", store_path: str = "data/index/docstore.json"):
        self.faiss_path = faiss_path
        self.store_path = store_path
        self.index = None                   # faiss.IndexIDMap2
        self.dim = 384
        self._store: Dict[str, Dict] = {}   # id (str) -> {"text": str, "source": str}
        self._next_id: int = 1
        self._loaded: bool = False

    # -------- persistence --------
    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)

    def save(self):
        self._ensure_dirs()
        if self.index is None:
            # create empty index if needed so files always exist
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base)
        faiss.write_index(self.index, self.faiss_path)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump({"next_id": self._next_id, "store": self._store}, f, ensure_ascii=False, indent=2)

    def load(self):
        # idempotent
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
        else:
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base)
        if os.path.exists(self.store_path):
            meta = json.load(open(self.store_path, "r", encoding="utf-8"))
            self._next_id = int(meta.get("next_id", 1))
            self._store = {str(k): v for k, v in meta.get("store", {}).items()}
        else:
            self._next_id = 1
            self._store = {}
        # infer dim if possible
        if self.index is not None and hasattr(self.index, "d"):
            self.dim = int(self.index.d)
        self._loaded = True

    def _lazy_load(self):
        if not self._loaded:
            self.load()

    # -------- building / adding / removing --------
    def build(self, X: np.ndarray, records: List[Dict]):
        """Rebuild from scratch."""
        self._lazy_load()
        if X is None or len(X.shape) != 2:
            X = np.zeros((0, self.dim), dtype="float32")
        if X.shape[0] == 0:
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base)
            self._store = {}
            self._next_id = 1
            self.save()
            return

        self.dim = X.shape[1]
        base = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap2(base)

        ids = np.arange(self._next_id, self._next_id + X.shape[0], dtype="int64")
        self.index.add_with_ids(X.astype("float32"), ids)

        # store
        self._store = {}
        for i, rid in enumerate(ids):
            rec = records[i]
            self._store[str(int(rid))] = {
                "text": rec.get("text") or "",
                "source": rec.get("source") or "",
            }
        self._next_id = int(ids[-1]) + 1
        self.save()

    def add(self, vecs: np.ndarray, records: List[Dict]):
        """Append new vectors + records (incremental)."""
        self._lazy_load()
        if vecs is None or vecs.shape[0] == 0:
            return {"added": 0}
        if self.index is None or self.index.d != vecs.shape[1]:
            # initialize a fresh map with correct dim if inconsistent
            base = faiss.IndexFlatIP(vecs.shape[1])
            self.index = faiss.IndexIDMap2(base)
            self._store = {}
            self._next_id = 1
        ids = np.arange(self._next_id, self._next_id + vecs.shape[0], dtype="int64")
        self.index.add_with_ids(vecs.astype("float32"), ids)
        for i, rid in enumerate(ids):
            rec = records[i]
            self._store[str(int(rid))] = {
                "text": rec.get("text") or "",
                "source": rec.get("source") or "",
            }
        self._next_id = int(ids[-1]) + 1
        self.save()
        return {"added": int(vecs.shape[0]), "first_id": int(ids[0]), "last_id": int(ids[-1])}

    def remove_sources(self, sources: List[str]) -> Dict:
        """Delete all vectors whose doc 'source' matches any of the given paths (exact match)."""
        self._lazy_load()
        if not sources:
            return {"removed": 0, "matched_ids": []}
        sources_set = {s.replace("\\", "/") for s in sources}
        # collect ids to remove
        to_remove: List[int] = []
        for sid, meta in list(self._store.items()):
            if meta.get("source", "").replace("\\", "/") in sources_set:
                to_remove.append(int(sid))
        if not to_remove:
            return {"removed": 0, "matched_ids": []}
        # remove from faiss
        arr = np.array(to_remove, dtype="int64")
        selector = faiss.IDSelectorArray(arr.size, faiss.swig_ptr(arr))
        removed = self.index.remove_ids(selector)
        # drop from store
        for rid in to_remove:
            self._store.pop(str(rid), None)
        self.save()
        # removed == number actually removed (faiss returns Int64)
        return {"removed": int(removed), "matched_ids": to_remove}

    # -------- queries --------
    def query_dense(self, qv: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        self._lazy_load()
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(qv.astype("float32"), k)
        out: List[Tuple[Dict, float]] = []
        for j in range(I.shape[1]):
            idx_id = int(I[0, j])
            if idx_id == -1:
                continue
            meta = self._store.get(str(idx_id))
            if not meta:
                continue
            out.append(({"text": meta.get("text", ""), "source": meta.get("source", "")}, float(D[0, j])))
        return out

    def query_sparse(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        # Placeholder for a TF-IDF layer if added later
        return []

    def query_hybrid(self, qv: np.ndarray, query: str, k: int = 5, alpha: float = 0.6) -> List[Tuple[Dict, float]]:
        # Currently return dense results; hook for future mixing
        return self.query_dense(qv, k)

    # -------- misc --------
    def size(self) -> int:
        self._lazy_load()
        # number of records in store (more robust than index.ntotal after selective deletes)
        return len(self._store)
