# app/rag/reranker.py
import os
import warnings
from typing import List, Dict, Tuple

# --- Quiet down tokenizers/transformers warnings (incl. clean_up_tokenization_spaces) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str):
        # CrossEncoder internally uses Transformers; verbosity has been reduced above.
        self.model = CrossEncoder(model_name, max_length=384)

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
        pairs = [(query, c.get("text", "")) for c in candidates]
        if not pairs:
            return []
        scores = self.model.predict(pairs, convert_to_numpy=True)
        order = scores.argsort()[::-1][:top_k]
        return [(candidates[i], float(scores[i])) for i in order]
