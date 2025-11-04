# app/rag/chunker.py
import re
from typing import List, Dict

def split_markdown(
    text: str,
    max_chars: int = 1200,
    overlap: int = 120,
    min_chars: int = 200,
) -> List[Dict]:
    """
    Split Markdown by headings and then into sliding windows with overlap.
    Ensures chunks are at least `min_chars`.
    """
    parts = re.split(r"\n(?=#+ )", text)
    chunks: List[Dict] = []
    for part in parts:
        start = 0
        n = len(part)
        while start < n:
            end = min(n, start + max_chars)
            chunk = part[start:end]
            if len(chunk) >= min_chars:
                chunks.append({"text": chunk.strip()})
            # advance with overlap
            if end >= n:
                break
            start = end - overlap if (end - overlap) > start else end
    return chunks
