# app/rag/index.py
import json, pathlib
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

class DocIndex:
    """
    Serves dense (FAISS), sparse (TF-IDF cosine), and hybrid retrieval over a JSON doc store.
    """
    def __init__(self, index_path: str, store_path: str):
        self.index_path = pathlib.Path(index_path)
        self.store_path = pathlib.Path(store_path)
        self.index = None
        self.store: List[Dict] = []
        self._tfidf = None
        self._tfidf_mat = None
        if self.index_path.exists() and self.store_path.exists():
            self._load()

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.store = json.loads(self.store_path.read_text(encoding="utf-8"))
        texts = [(rec.get("text") or "") for rec in self.store]
        self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1, 2))
        # CSR matrix
        self._tfidf_mat = self._tfidf.fit_transform(texts)

    def build(self, X: np.ndarray, records: List[Dict]):
        dim = X.shape[1] if X.size else 384
        self.index = faiss.IndexFlatIP(dim)
        if X.size:
            self.index.add(X.astype(np.float32))
        self.store = records
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.store_path.write_text(json.dumps(self.store, ensure_ascii=False), encoding="utf-8")
        texts = [(rec.get("text") or "") for rec in records]
        self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1, 2))
        self._tfidf_mat = self._tfidf.fit_transform(texts)

    def query_dense(self, qvecs: np.ndarray, k: int = 6) -> List[Tuple[Dict, float]]:
        if self.index is None or len(self.store) == 0:
            return []
        k = max(1, min(k, len(self.store)))
        D, I = self.index.search(qvecs.astype(np.float32), k)
        res = []
        for j in range(I.shape[1]):
            idx = int(I[0, j])
            if 0 <= idx < len(self.store):
                res.append((self.store[idx], float(D[0, j])))
        return res

    def query_sparse(self, query: str, k: int = 6) -> List[Tuple[Dict, float]]:
        """
        TF-IDF cosine scores. Use .toarray() instead of .A (which doesn't exist on csr_matrix).
        """
        if self._tfidf is None or self._tfidf_mat is None or len(self.store) == 0:
            return []
        k = max(1, min(k, len(self.store)))

        # Transform query -> CSR
        q = self._tfidf.transform([query])

        # L2-normalize both sides; sk_normalize preserves CSR
        qn = sk_normalize(q)
        dn = sk_normalize(self._tfidf_mat)

        # Cosine scores = dot product of normalized vectors (1 x N sparse) -> dense row
        scores = (qn @ dn.T).toarray().ravel()

        # Top-k indices
        idxs = np.argsort(scores)[::-1][:k]
        return [(self.store[int(i)], float(scores[int(i)])) for i in idxs]

    def query_hybrid(self, qvecs: np.ndarray, query: str, k: int = 6, alpha: float = 0.6) -> List[Tuple[Dict, float]]:
        if len(self.store) == 0:
            return []

        # Pull a bit more from each side to blend
        kk = max(1, min(max(2 * k, 50), len(self.store)))
        dense = self.query_dense(qvecs, k=kk)
        sparse = self.query_sparse(query, k=kk)

        # Key by (source, first 64 chars) to dedup while keeping stability
        def key(rec): 
            return (rec.get("source", ""), (rec.get("text", "") or "")[:64])

        dmap = {key(r): s for r, s in dense}
        smap = {key(r): s for r, s in sparse}

        seen = {}
        for rec, sc in dense + sparse:
            kx = key(rec)
            d = dmap.get(kx, 0.0)
            s = smap.get(kx, 0.0)
            score = float(alpha) * float(d) + float(1.0 - alpha) * float(s)
            if kx not in seen or score > seen[kx][1]:
                seen[kx] = (rec, score)

        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:k]
        return ranked
