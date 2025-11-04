import json, pathlib
from glob import glob
from app.rag.chunker import split_markdown
from app.rag.embedder import Embedder
from app.rag.index import DocIndex


import yaml
cfg = yaml.safe_load(open("configs/settings.yaml"))


PROC = pathlib.Path("data/processed")
files = sorted(glob(str(PROC / "*.md")))


records = []
for fp in files:
text = pathlib.Path(fp).read_text(encoding="utf-8")
for ch in split_markdown(text, **cfg["chunk"]):
records.append({
"text": ch["text"],
"source": fp,
})


embedder = Embedder(cfg["retrieval"]["embedder_model"], cfg["retrieval"]["normalize"])
X = embedder.encode([r["text"] for r in records])


idx = DocIndex(cfg["retrieval"]["faiss_index"], cfg["retrieval"]["store_json"])
idx.build(X, records)
print(f"Indexed {len(records)} chunks")