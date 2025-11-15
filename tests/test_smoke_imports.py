# tests/test_smoke_imports.py
"""
Very light smoke tests that ensure core modules import without error.
This catches missing dependencies or syntax errors early.
"""

def test_import_core_modules():
    import app.streamlit_app  # noqa: F401
    import app.rag.utils  # noqa: F401
    import app.rag.embedder  # noqa: F401
    import app.rag.index  # noqa: F401
    import app.rag.chunker  # noqa: F401
    import app.llm.answerer  # noqa: F401
