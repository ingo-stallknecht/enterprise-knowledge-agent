# app/rag/index.py
import json, pathlib
import faiss
import numpy as np


class DocIndex:
def __init__(self, index_path: str, store_path: str):
self.index_path = pathlib.Path(index_path)
self.store_path = pathlib.Path(store_path)
self.index = None
self.store = []


def build(self, embeddings: np.ndarray, metadata_list):
d = embeddings.shape[1]
self.index = faiss.IndexFlatIP(d)
self.index.add(embeddings.astype("float32"))
self.store = metadata_list
self.save()


def save(self):
self.index_path.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(self.index, str(self.index_path))
self.store_path.write_text(json.dumps(self.store, ensure_ascii=False, indent=2), encoding="utf-8")


def load(self):
self.index = faiss.read_index(str(self.index_path))
self.store = json.loads(self.store_path.read_text(encoding="utf-8"))
return self


def search(self, query_vec: np.ndarray, k: int):
scores, idxs = self.index.search(query_vec.astype("float32"), k)
idxs = idxs[0].tolist(); scores = scores[0].tolist()
results = []
for i, s in zip(idxs, scores):
if i < 0: continue
meta = self.store[i]
meta["score"] = float(s)
results.append(meta)
return results