# app/rag/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.
    Encodes a list of texts into embeddings (optionally normalized).
    """

    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        X = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        if self.normalize:
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X
