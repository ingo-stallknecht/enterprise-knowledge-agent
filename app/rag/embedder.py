from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
def __init__(self, model_name: str, normalize: bool = True):
self.model = SentenceTransformer(model_name)
self.normalize = normalize
def encode(self, texts):
vecs = self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
if self.normalize:
norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
vecs = vecs / norms
return vecs