# scripts/build_index.py
import json
import pathlib
from glob import glob
from typing import Callable, Dict, Optional, List

from app.rag.chunker import split_markdown
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import load_cfg

def build_index(progress_cb: Optional[Callable[[str, Dict], None]] = None) -> Dict:
    """
    Build FAISS index with optional progress callbacks.

    Events emitted (with payloads):
      - "scan_total": {"files": <int>}
      - "scan_file":  {"i": <int>, "n": <int>, "file": <str>}
      - "chunk_progress": {"files_done": <int>, "files_total": <int>, "chunks": <int>, "file": <str>}
      - "embed_progress": {"batch_idx": <int>, "batches": <int>, "rows_done": <int>, "rows_total": <int>, "batch_size": <int>}
      - "write_index": {}
      - "done": {"num_files": <int>, "num_chunks": <int>}
    """
    def emit(ev, pl=None):
        if progress_cb: progress_cb(ev, pl or {})

    CFG = load_cfg("configs/settings.yaml")
    PROC = pathlib.Path("data/processed")
    files: List[str] = sorted(glob(str(PROC / "**/*.md"), recursive=True))
    emit("scan_total", {"files": len(files)})

    records: List[Dict] = []
    for i, fp in enumerate(files, start=1):
        emit("scan_file", {"i": i, "n": len(files), "file": fp})
        text = pathlib.Path(fp).read_text(encoding="utf-8")
        chunks = list(split_markdown(text, **(CFG.get("chunk") or {})))
        for ch in chunks:
            records.append({"text": ch["text"], "source": fp})
        emit("chunk_progress", {
            "files_done": i,
            "files_total": len(files),
            "chunks": len(records),
            "file": fp
        })

    from numpy import vstack
    import numpy as np
    idx = DocIndex(CFG["retrieval"]["faiss_index"], CFG["retrieval"]["store_json"])

    if not records:
        X = np.zeros((0, 384), dtype="float32")
        idx.build(X, [])
        emit("done", {"num_files": len(files), "num_chunks": 0})
        return {"num_files": len(files), "num_chunks": 0}

    embedder = Embedder(CFG["retrieval"]["embedder_model"], CFG["retrieval"]["normalize"])
    texts = [r["text"] for r in records]

    batch_size = 64
    vecs = []
    rows_total = len(texts)
    batches = max(1, (rows_total + batch_size - 1) // batch_size)
    for b in range(batches):
        s = b * batch_size
        e = min(rows_total, s + batch_size)
        vecs.append(embedder.encode(texts[s:e]))
        emit("embed_progress", {
            "batch_idx": b + 1,
            "batches": batches,
            "rows_done": e,
            "rows_total": rows_total,
            "batch_size": batch_size
        })
    X = vstack(vecs)

    emit("write_index", {})
    idx.build(X, records)

    stats = {"num_files": len(files), "num_chunks": len(records)}
    emit("done", stats)
    return stats

if __name__ == "__main__":
    def printer(ev, pl):
        if ev == "scan_total":
            print(f"[scan] files={pl.get('files')}")
        elif ev == "scan_file":
            print(f"[scan] {pl.get('i')}/{pl.get('n')}  {pl.get('file')}")
        elif ev == "chunk_progress":
            print(f"[chunk] files {pl.get('files_done')}/{pl.get('files_total')}  chunks={pl.get('chunks')}  file={pl.get('file')}")
        elif ev == "embed_progress":
            print(f"[embed] batch {pl.get('batch_idx')}/{pl.get('batches')} rows {pl.get('rows_done')}/{pl.get('rows_total')} (bs={pl.get('batch_size')})")
        elif ev == "write_index":
            print("[write] writing index/store")
        elif ev == "done":
            print(f"[done] files={pl.get('num_files')} chunks={pl.get('num_chunks')}")
    stats = build_index(progress_cb=printer)
    print(stats)
