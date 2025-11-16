# tests/test_retriever.py
from app.rag.retriever import Retriever


def test_dense_returns_at_most_k():
    docs = [{"text": f"doc-{i}", "source": f"src-{i}"} for i in range(10)]
    r = Retriever(docs)

    out = r.dense("any query", k=3)
    assert len(out) == 3
    assert out[0]["text"] == "doc-0"
    assert out[1]["source"] == "src-1"
