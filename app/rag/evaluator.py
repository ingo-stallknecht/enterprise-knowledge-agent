# app/rag/evaluator.py
"""
Tiny evaluation helper used by tests and local experiments.

This is deliberately minimal: it provides a Recall@k style metric on top
of a Retriever-like object with a ``dense(query, k)`` method.
"""

from typing import Dict, List


class Evaluator:
    def __init__(self, retriever, k: int):
        """
        Parameters
        ----------
        retriever:
            Any object with a ``dense(query: str, k: int) -> List[Dict]`` method.
        k:
            Number of top documents to consider for recall@k.
        """
        self.retriever = retriever
        self.k = k

    def recall_at_k(self, query: str, relevant_sources: List[str]) -> float:
        """Compute a simple recall@k on source ids.

        Parameters
        ----------
        query:
            The query string.
        relevant_sources:
            List of source identifiers (e.g. file paths) that are considered relevant.

        Returns
        -------
        float
            Fraction of relevant sources that appear among the top-k results.
        """
        if not relevant_sources:
            return 0.0

        results: List[Dict] = self.retriever.dense(query, k=self.k)
        found = {r.get("source") for r in results}
        hits = sum(1 for src in relevant_sources if src in found)
        return hits / len(relevant_sources)
