"""
Build FAISS index with optional progress callbacks.

This script:
- Scans all Markdown files under data/processed/
- Splits them into chunks using the shared chunker
- Embeds all chunks with the configured SentenceTransformers model
- Builds a fresh FAISS index + docstore.json

Supports progress callbacks for Airflow / CLI visualization.
"""

from __future__ import annotations

import sys
import pathlib
import warnings
from glob import glob
from typing import Callable, Dict, Optional, List

# --- Suppress transformers FutureWarning about clean_up_tokenization_spaces ---
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)

# Make project root importable when called as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# App imports (after sys.path adjustment)
from app.rag.chunker import split_markdown
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import load_cfg


def build_index(progress_cb: Optional[Callable[[str, Dict], None]] = None) -> Dict:
    """
    Build FAISS index with optional progress callbacks.

    Progress events emitted:
      - "scan_total": {"files": <int>}
      - "scan_file": {"file": <str>, "i": <int>, "n": <int>}
      - "chunk_progress": {"files_done": <int>, "files_total": <int>, "chunks": <int>}
      - "embed_progress": {"batch_idx": <int>, "batches": <int>, "rows_done": <int>, "rows_total": <int>}
      - "write_index": {}
      - "done": {"num_files": <int>, "num_chunks": <int>}
    """

    def emit(event: str, payload: Optional[Dict] = None) -> None:
        if progress_cb:
            progress_cb(event, payload or {})

    CFG = load_cfg("configs/settings.yaml")
    PROC = pathlib.Path("data/processed")

    # Collect all markdown files
    files: List[str] = sorted(glob(str(PROC / "**/*.md"), recursive=True))
    emit("scan_total", {"files": len(files)})

    records: List[Dict] = []

    # Step 1: Chunk extraction
    for i, fp in enumerate(files, start=1):
        emit("scan_file", {"file": fp, "i": i, "n": len(files)})

        text = pathlib.Path(fp).read_text(encoding="utf-8", errors="ignore")
        chunks = split_markdown(text, **(CFG.get("chunk") or {}))

        for ch in chunks:
            records.append(
                {
                    "text": ch["text"],
                    "source": fp.replace("\\", "/"),
                }
            )

        emit(
            "chunk_progress",
            {
                "files_done": i,
                "files_total": len(files),
                "chunks": len(records),
            },
        )

    # Prepare index
    idx = DocIndex(
        CFG["retrieval"]["faiss_index"],
        CFG["retrieval"]["store_json"],
    )

    # If no records, create empty index
    if not records:
        import numpy as np

        X = np.zeros((0, 384), dtype="float32")
        idx.build(X, [])
        emit("done", {"num_files": len(files), "num_chunks": 0})
        return {"num_files": len(files), "num_chunks": 0}

    # Step 2: Embeddings
    embed_cfg = CFG["retrieval"]
    embedder = Embedder(
        embed_cfg["embedder_model"],
        embed_cfg["normalize"],
    )

    texts = [r["text"] for r in records]
    rows_total = len(texts)
    batch_size = 64
    batches = max(1, (rows_total + batch_size - 1) // batch_size)

    vecs = []
    for b in range(batches):
        start = b * batch_size
        end = min(rows_total, start + batch_size)
        batch_vecs = embedder.encode(texts[start:end])
        vecs.append(batch_vecs)

        emit(
            "embed_progress",
            {
                "batch_idx": b + 1,
                "batches": batches,
                "rows_done": end,
                "rows_total": rows_total,
            },
        )

    # Stack vectors
    from numpy import vstack

    X = vstack(vecs)

    # Step 3: Write index
    emit("write_index", {})
    idx.build(X, records)

    stats = {"num_files": len(files), "num_chunks": len(records)}
    emit("done", stats)
    return stats


if __name__ == "__main__":

    def printer(event: str, payload: Dict) -> None:
        """Pretty logging when run from CLI."""
        if event == "scan_total":
            print(f"[scan] files={payload.get('files')}")
        elif event == "scan_file":
            print(f"[scan] {payload.get('i')}/{payload.get('n')} {payload.get('file')}")
        elif event == "chunk_progress":
            print(
                f"[chunk] {payload.get('files_done')}/{payload.get('files_total')} "
                f"chunks={payload.get('chunks')}"
            )
        elif event == "embed_progress":
            print(
                f"[embed] batch {payload.get('batch_idx')}/{payload.get('batches')} "
                f"rows={payload.get('rows_done')}/{payload.get('rows_total')}"
            )
        elif event == "write_index":
            print("[write] writing index/store")
        elif event == "done":
            print(f"[done] files={payload.get('num_files')} " f"chunks={payload.get('num_chunks')}")

    result = build_index(progress_cb=printer)
    print(result)
