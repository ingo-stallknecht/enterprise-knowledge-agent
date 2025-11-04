# scripts/build_index.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pathlib
from glob import glob
import yaml

from app.rag.chunker import split_markdown
from app.rag.embedder import Embedder
from app.rag.index import DocIndex

# Load config with safe defaults
cfg = yaml.safe_load(open("configs/settings.yaml", encoding="utf-8"))

chunk_cfg = (cfg.get("chunk") or {})
max_chars  = int(chunk_cfg.get("max_chars", 1200))
overlap    = int(chunk_cfg.get("overlap", 120))
min_chars  = int(chunk_cfg.get("min_chars", 200))

PROC = pathlib.Path("data/processed")
files = sorted(glob(str(PROC / "*.md")))
if not files:
    raise SystemExit("No processed markdown files found in data/processed. Run fetch script first.")

records = []
for fp in files:
    text = pathlib.Path(fp).read_text(encoding="utf-8", errors="ignore")
    for ch in split_markdown(text, max_chars=max_chars, overlap=overlap, min_chars=min_chars):
        records.append({"text": ch["text"], "source": fp})

if not records:
    raise SystemExit("No chunks produced. Check chunk sizes or input markdown.")

# Build embeddings
embedder = Embedder(
    cfg.get("retrieval", {}).get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"),
    bool(cfg.get("retrieval", {}).get("normalize", True)),
)
X = embedder.encode([r["text"] for r in records])

# Build index
faiss_index_path = cfg.get("retrieval", {}).get("faiss_index", "data/index/handbook.index")
store_json_path  = cfg.get("retrieval", {}).get("store_json", "data/index/docstore.json")

idx = DocIndex(faiss_index_path, store_json_path)
idx.build(X, records)

print(f"Indexed {len(records)} chunks")
