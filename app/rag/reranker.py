# app/rag/reranker.py
from typing import List, Dict, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class Reranker:
    """Cross-encoder reranker. Falls back to a simple heuristic if model unavailable."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if CrossEncoder:
            try:
                # Let sentence-transformers use HF cache via env vars; avoid unsupported kwargs.
                self.model = CrossEncoder(model_name)
            except Exception:
                self.model = None

    def rerank(
        self, query: str, candidates: List[Dict], top_k: int = 6
    ) -> List[Tuple[Dict, float]]:
        if not candidates:
            return []

        # Fallback heuristic when CrossEncoder isn't available:
        if not self.model:
            # score ~ normalized length up to 1000 chars (simple, deterministic)
            scored = [(c, float(min(len(c.get("text") or ""), 1000)) / 1000.0) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        pairs = [(query, (c.get("text") or "")) for c in candidates]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
