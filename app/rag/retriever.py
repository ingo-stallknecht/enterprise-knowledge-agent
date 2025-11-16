# app/rag/retriever.py
"""
Lightweight Retriever used by unit tests and legacy scripts.

The Streamlit app uses Embedder + DocIndex directly. This module provides
a tiny, dependency-free retriever for simple tests.
"""

from typing import Dict, List


class Retriever:
    """In-memory toy retriever.

    Parameters
    ----------
    documents:
        List of dicts with at least keys ``text`` and ``source``.
    """

    def __init__(self, documents: List[Dict]):
        self._docs = documents or []

    def dense(self, query: str, k: int = 5) -> List[Dict]:
        """Return at most k documents.

        This is intentionally simple: it ignores the query and just returns
        the first k docs. For unit tests we only care about shape and
        determinism, not semantic quality.
        """
        _ = query  # keep signature, avoid unused warning
        return self._docs[:k]
