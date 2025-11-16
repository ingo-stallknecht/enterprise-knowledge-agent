# app/rag/embedder.py
from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Thin wrapper around SentenceTransformer used throughout the app.

    - Provides a default model so tests can call Embedder() with no args.
    - Returns float32 numpy arrays.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        # device="cpu" is fine for Streamlit Cloud and CI
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings into a 2D float32 numpy array."""
        if not texts:
            # Return an empty (0, d) matrix if no texts are given
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")

        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype("float32")
