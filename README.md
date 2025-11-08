# Enterprise Knowledge Agent

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enterprise-knowledge-agent-6fs3f3ha2qmmbi4wwgxyjn.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/ingo-stallknecht/enterprise-knowledge-agent)

Hybrid RAG + reranker + GPT-optional answers with a polished Streamlit UI, async uploads → reindex pipeline,
evaluation + MLflow logging, and threshold-gated promotion/true rollback of the FAISS index snapshot.

**What**: Local, production-style RAG agent over GitLab Handbook with agent actions, Airflow weekly refresh, and MLflow-gated promotion + rollback.

---

## Features
- RAG with FAISS + SentenceTransformers
- Actions: `retrieve`, `answer_with_citations`, `upsert_wiki_page`, `create_ticket`
- FastAPI server with OpenAPI docs
- Airflow DAG for weekly ingest → index → eval → promote
- MLflow metrics + model registry stages (Production, Archived)
- Tests & Makefile
- Fully local, cloud-free

---

## Quickstart (Streamlit demo)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_mono.py
