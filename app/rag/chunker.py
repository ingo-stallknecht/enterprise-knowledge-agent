# app/rag/chunker.py
from __future__ import annotations

import re
from typing import Dict, List

# Very lightweight markdown chunker:
# - Keeps headings together with their following paragraph(s)
# - Splits into chunks of at most `max_chars`
# - Uses a simple sliding-window when a single block is very long


def _normalize(text: str) -> str:
    """Normalize newlines and strip outer whitespace."""
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _merge_headings_with_blocks(blocks: List[str]) -> List[str]:
    """Attach heading-only blocks to the following block so they stay together."""
    merged: List[str] = []
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        if block.startswith("#") and i + 1 < len(blocks):
            # Merge heading with next block
            next_block = blocks[i + 1].strip()
            merged.append(f"{block}\n\n{next_block}")
            i += 2
        else:
            if block:
                merged.append(block)
            i += 1
    return merged


def _split_long_block(block: str, max_chars: int, overlap: int) -> List[str]:
    """Split a single long block into overlapping windows."""
    text = block.strip()
    if not text:
        return []

    # Ensure we don't end up with non-positive step
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - min(overlap, max_chars // 2)

    return chunks


def split_markdown(
    text: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> List[Dict]:
    """
    Split a markdown document into chunks.

    Each returned chunk is a dict with at least the key:
        - "text": the chunk text

    For non-empty input, this always returns at least one chunk.
    """
    norm = _normalize(text)
    if not norm:
        return []

    # First pass: logical blocks separated by blank lines
    raw_blocks = re.split(r"\n\s*\n", norm)
    raw_blocks = [b for b in (blk.strip() for blk in raw_blocks) if b]

    if not raw_blocks:
        # Fallback: treat entire text as one block
        raw_blocks = [norm]

    # Merge "# Heading" blocks with their following paragraph
    blocks = _merge_headings_with_blocks(raw_blocks)

    chunks: List[Dict] = []
    current: str = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # If the block itself is huge, split it into windows first
        if len(block) > max_chars:
            # Flush any current chunk
            if current:
                chunks.append({"text": current.strip()})
                current = ""

            long_parts = _split_long_block(block, max_chars=max_chars, overlap=overlap)
            for part in long_parts:
                chunks.append({"text": part})
            continue

        if not current:
            current = block
        elif len(current) + 2 + len(block) <= max_chars:
            # 2 accounts for "\n\n" joiner
            current = f"{current}\n\n{block}"
        else:
            # Flush current and start new chunk
            chunks.append({"text": current.strip()})
            current = block

    if current.strip():
        chunks.append({"text": current.strip()})

    # Final safety: for non-empty input we must return at least one chunk
    if not chunks and norm:
        chunks = [{"text": norm}]

    return chunks
