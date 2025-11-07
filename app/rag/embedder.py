# app/rag/embedder.py
import os
import warnings
from typing import List
import numpy as np

# --- Quiet transformers warnings ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
warnings.filterwarnings(
    "ignore",
    message=r"`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
    module=r".*transformers.*",
)

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, normalize: bool = True, progress: bool | None = None):
        """
        progress: if None, read from env EKA_PROGRESS (1/0). Default False.
        """
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        if progress is None:
            self.progress = os.getenv("EKA_PROGRESS", "0").strip() in ("1", "true", "True")
        else:
            self.progress = bool(progress)

    def encode(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=self.progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return X

