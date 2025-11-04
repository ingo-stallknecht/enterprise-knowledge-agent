# app/rag/index.py
import json
import pathlib
import numpy as np

# Try FAISS; fall back to sklearn on Windows if needed
try:
    import faiss
    USE_FAISS = True
except ImportError:
    from sklearn.metrics.pairwise import cosine_similarity
    USE_FAISS = False


class DocIndex:
    """
    Handles building and querying of the vector index.
    Uses FAISS if available, otherwise sklearn cosine similarity.
    """

    def __init__(self, index_path: str, store_path: str):
        self.index_path = pathlib.Path(index_path)
        self.store_path = pathlib.Path(store_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, X: np.ndarray, records: list[dict]):
        """Builds and saves the FAISS (or sklearn) index and the store."""
        if USE_FAISS:
            dim = X.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
            faiss.normalize_L2(X)
            index.add(X)
            faiss.write_index(index, str(self.index_path))
        else:
            # fallback: store embeddings directly for sklearn cosine search
            np.save(str(self.index_path.with_suffix(".npy")), X)

        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def query(self, query_vec: np.ndarray, k: int = 5):
        """Return top-k most similar documents for a query vector."""
        if USE_FAISS:
            index = faiss.read_index(str(self.index_path))
            faiss.normalize_L2(query_vec)
            D, I = index.search(query_vec, k)
            with open(self.store_path, encoding="utf-8") as f:
                store = json.load(f)
            return [(store[i], float(D[0][j])) for j, i in enumerate(I[0])]
        else:
            X = np.load(str(self.index_path.with_suffix(".npy")))
            sims = cosine_similarity(query_vec, X)[0]
            topk_idx = np.argsort(sims)[::-1][:k]
            with open(self.store_path, encoding="utf-8") as f:
                store = json.load(f)
            return [(store[i], float(sims[i])) for i in topk_idx]
