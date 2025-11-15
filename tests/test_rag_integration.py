# tests/test_rag_integration.py
"""
Simple integration-style test: chunk -> embed -> index -> query.

This does not touch Streamlit, but exercises the core RAG pipeline end-to-end
on a tiny in-memory corpus.
"""

from app.rag.chunker import split_markdown
from app.rag.embedder import Embedder
from app.rag.index import DocIndex


def test_rag_pipeline_simple_query(tmp_path):
    # 1) Create a tiny markdown "document"
    md_text = """
# Values in Performance Reviews

We emphasize iteration and small, frequent changes.
Performance reviews reference concrete behaviors, not vague opinions.

## Feedback

Feedback should be time-boxed. Colleagues collect input before the decision deadline.
"""
    chunks = split_markdown(md_text, max_chars=400, overlap=50)
    assert chunks, "Chunking should produce at least one chunk."
    texts = [c["text"] for c in chunks]

    # 2) Embed chunks
    emb = Embedder()
    X = emb.encode(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] > 0

    # 3) Build a temporary index on disk (in a temp folder)
    faiss_path = tmp_path / "test.index"
    store_path = tmp_path / "test_docstore.json"
    idx = DocIndex(faiss_path=str(faiss_path), store_path=str(store_path))

    records = [{"text": t, "source": "test_doc.md"} for t in texts]
    idx.build(X, records)
    assert idx.size() == len(records)

    # 4) Query with a relevant question and verify we get something meaningful back
    q_vec = emb.encode(["How is feedback time-boxed in reviews?"])
    results = idx.query_dense(q_vec, k=3)
    assert results, "Expected at least one retrieval result."

    top_record, score = results[0]
    assert "feedback" in top_record["text"].lower() or "time" in top_record["text"].lower()
    assert score > 0.0
