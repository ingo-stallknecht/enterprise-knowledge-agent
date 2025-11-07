# Enterprise Knowledge Agent

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-USERNAME-enterprise-knowledge-agent.streamlit.app)

Hybrid RAG + reranker + GPT-optional answers with a polished Streamlit UI, async uploads → reindex pipeline,
evaluation + MLflow logging, and threshold-gated promotion/true rollback of the FAISS index snapshot.



**What**: Local, production‑style RAG agent over GitLab Handbook with agent actions, Airflow weekly refresh, and MLflow‑gated promotion + rollback.


## Features
- RAG with FAISS + SentenceTransformers
- Actions: `retrieve`, `answer_with_citations`, `upsert_wiki_page`, `create_ticket`
- FastAPI server with OpenAPI docs
- Airflow DAG for weekly ingest → index → eval → promote
- MLflow metrics + model registry stages (Production, Archived)
- Tests & Makefile
- Fully local, cloud‑free


## Quickstart
```bash
cp .env.example .env
make init
make pipeline
make api # http://127.0.0.1:8000/docs