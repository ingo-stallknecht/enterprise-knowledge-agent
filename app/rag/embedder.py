# app/rag/embedder.py
from typing import List
import os
import inspect

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Thin wrapper around SentenceTransformer to:
    - Respect an optional HF cache directory (HF_HOME / TRANSFORMERS_CACHE).
    - Work with both older and newer sentence-transformers versions by
      only passing `trust_remote_code` if the library supports it.
    - Always return float32 numpy arrays and optionally normalize embeddings.
    """

    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize

        # Decide where to cache models (keeps Streamlit Cloud happy)
        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or None
        )

        # Build kwargs in a version-safe way
        extra_kwargs = {}
        try:
            sig = inspect.signature(SentenceTransformer.__init__)
            if "trust_remote_code" in sig.parameters:
                # Newer sentence-transformers / transformers support this flag
                extra_kwargs["trust_remote_code"] = False
        except Exception:
            # If anything goes wrong, just don't pass extra kwargs
            extra_kwargs = {}

        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                **extra_kwargs,
            )
        except TypeError:
            # Fallback: if even with inspection something goes wrong,
            # try again without any extra kwargs.
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Returns:
            np.ndarray of shape (n_texts, dim) with dtype float32.
        """
        if not texts:
            # Default dim 384 for MiniLM; downstream code can handle empty arrays.
            return np.zeros((0, 384), dtype="float32")

        # sentence-transformers already returns numpy if convert_to_numpy=True,
        # but we defensively convert and normalize.
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # we handle normalization ourselves
        )

        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)

        emb = emb.astype("float32")

        if self.normalize and emb.size > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms

        return emb
