# app/rag/index.py
import json, pathlib, os, math, time
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

def _file_mtime_days(path: str) -> Optional[float]:
    try:
        ts = os.path.getmtime(path)
        age_days = max(0.0, (time.time() - ts) / 86400.0)
        return age_days
    except Exception:
        return None

class DocIndex:
    """
    Serves dense (FAISS), sparse (TF-IDF cosine), and hybrid retrieval over a JSON doc store.
    Supports source priors and recency boosts for better inclusion of fresh/wiki content.
    """
    def __init__(
        self,
        index_path: str,
        store_path: str,
        *,
        wiki_boost: float = 0.0,
        recency_half_life_days: Optional[float] = None,
        recency_boost: float = 0.0,
    ):
        self.index_path = pathlib.Path(index_path)
        self.store_path = pathlib.Path(store_path)
        self.index = None
        self.store: List[Dict] = []
        self._tfidf = None
        self._tfidf_mat = None

        # priors
        self.wiki_boost = float(wiki_boost or 0.0)
        self.recency_half_life_days = float(recency_half_life_days) if recency_half_life_days else None
        self.recency_boost = float(recency_boost or 0.0)

        if self.index_path.exists() and self.store_path.exists():
            self._load()

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.store = json.loads(self.store_path.read_text(encoding="utf-8"))
        # build sparse matrix for query_sparse/hybrid
        texts = [(rec.get("text") or "") for rec in self.store]
        self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1,2))
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
        # rebuild tf-idf
        texts = [(rec.get("text") or "") for rec in records]
        self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1,2))
        self._tfidf_mat = self._tfidf.fit_transform(texts)

    # ---- Dense ----
    def query_dense(self, qvecs: np.ndarray, k: int = 6) -> List[Tuple[Dict, float]]:
        if self.index is None or len(self.store) == 0:
            return []
        D, I = self.index.search(qvecs.astype(np.float32), k)
        res = []
        for j in range(I.shape[1]):
            idx = int(I[0, j])
            if 0 <= idx < len(self.store):
                res.append((self.store[idx], float(D[0, j])))
        return res

    # ---- Sparse (TF-IDF cosine) ----
    def query_sparse(self, query: str, k: int = 6) -> List[Tuple[Dict, float]]:
        if self._tfidf is None or self._tfidf_mat is None or len(self.store) == 0:
            return []
        q = self._tfidf.transform([query])
        # cosine via normalized dot
        qn = sk_normalize(q)
        dn = sk_normalize(self._tfidf_mat)
        scores = (qn @ dn.T).A[0]
        idxs = np.argsort(scores)[::-1][:k]
        return [(self.store[i], float(scores[i])) for i in idxs]

    # ---- Priors ----
    def _prior_for_source(self, source_path: str) -> float:
        prior = 0.0
        sp = source_path.replace("\\", "/")
        if self.wiki_boost and "/data/processed/wiki/" in f"/{sp}":
            prior += self.wiki_boost
        if self.recency_half_life_days and self.recency_boost:
            age = _file_mtime_days(source_path)
            if age is not None:
                # Exponential decay: boost * 2^(-age/half_life)
                factor = 2.0 ** (-(age / max(0.1, self.recency_half_life_days)))
                prior += self.recency_boost * float(factor)
        return float(prior)

    # ---- Hybrid ----
    def query_hybrid(self, qvecs: np.ndarray, query: str, k: int = 6, alpha: float = 0.6) -> List[Tuple[Dict, float]]:
        # get wider candidate pools so priors have a chance to surface recency/wiki docs
        dense = self.query_dense(qvecs, k=max(2*k, 50))
        sparse = self.query_sparse(query, k=max(2*k, 50))
        # merge by doc identity
        def key(rec: Dict) -> str:
            return (rec.get("source","") + "|" + (rec.get("text","")[:64]))

        dmap, smap = {}, {}
        for rec, sc in dense: dmap[key(rec)] = sc
        for rec, sc in sparse: smap[key(rec)] = sc

        seen: Dict[str, Tuple[Dict, float]] = {}
        for rec, sc in dense + sparse:
            kx = key(rec)
            if kx in seen: 
                continue
            d = float(dmap.get(kx, 0.0))
            s = float(smap.get(kx, 0.0))
            base = alpha * d + (1.0 - alpha) * s
            base += self._prior_for_source(rec.get("source",""))
            seen[kx] = (rec, base)

        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:k]
        return ranked
