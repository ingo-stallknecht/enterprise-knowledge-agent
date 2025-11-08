# Enterprise Knowledge Agent

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_bage_black_white.svg)](https://enterprise-knowledge-agent-6fs3f3ha2qmmbi4wwgxyjn.streamlit.app/)
![Last Commit](https://img.shields.io/github/last-commit/ingo-stallknecht/enterprise-knowledge-agent)

Hybrid RAG with FAISS + SentenceTransformers, optional GPT-assisted answers, and a polished Streamlit UI.
Runs **locally** or on **Streamlit Cloud** with incremental indexing, safe wiki actions, and citations.

---

## What it is
A Streamlit-only knowledge app over a (GitLab-inspired) handbook + your uploads:
- Ask questions → retrieves supporting passages → answers with citations.
- Agent can:
  1) **Create** a wiki page (with confirmation, safety checks, and GPT-4 drafting when available),
  2) **Delete** wiki pages (restricted to `data/processed/wiki/`, confirmation required),
  3) Otherwise **answer** helpfully with no side effects.
- Indexing is **incremental** (no full re-embed on deletes) with a **prebuilt index** option for fast cold starts.

---

## Features
- **Retrieval:** FAISS dense (MiniLM), optional reranker
- **Answering:** Extractive fallback; GPT-assisted when `OPENAI_API_KEY` is set
- **Agent actions:** Create/Delete wiki (confirm first) + normal answers
- **Indexing:** Incremental add; delete only rebuilds FAISS from cached vectors
- **Prebuilt index:** Drop files into `data/index/prebuilt/` to skip first-run builds
- **Safety:** Blocks harmful titles/content on wiki creation; deletes limited to wiki folder
- **UX:** Context visualizer (sentence → strongest source), citations with previews

---

## Quickstart (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_mono.py
