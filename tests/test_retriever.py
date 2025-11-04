# tests/test_retriever.py
import pathlib, yaml
from app.rag.retriever import Retriever


def test_retriever_loads():
cfg = yaml.safe_load(open("configs/settings.yaml"))
# requires index to exist (run make index once for CI demo)
r = Retriever(cfg["retrieval"]["embedder_model"], cfg["retrieval"]["normalize"], cfg["retrieval"]["faiss_index"], cfg["retrieval"]["store_json"])
res = r.retrieve("values", k=2)
assert isinstance(res, list)