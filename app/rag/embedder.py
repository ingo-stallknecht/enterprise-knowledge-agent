# app/rag/embedder.py
from typing import List
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str, normalize: bool = True):
        # Use a writable cache folder (works on Streamlit Cloud)
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        # Explicitly disable any auth/token usage
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            use_auth_token=False,           # <- never read token
            trust_remote_code=False
        )
        self.normalize = normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return X
