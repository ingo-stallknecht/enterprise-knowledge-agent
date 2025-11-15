# Enterprise Knowledge Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![CI](https://github.com/ingo-stallknecht/enterprise-knowledge-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ingo-stallknecht/enterprise-knowledge-agent/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enterprise-knowledge-agent-yjmgeytutbelggasr2qbzy.streamlit.app/)
![Last Commit](https://img.shields.io/github/last-commit/ingo-stallknecht/enterprise-knowledge-agent)

---

## Overview

The **Enterprise Knowledge Agent (EKA)** is a Retrieval-Augmented Generation (RAG) and knowledge-editing system.  
It ingests Markdown documents, builds semantic search using embeddings and FAISS, answers questions with citations, and provides safe tools to create or edit internal wiki pages.

The system runs:

- **Locally:** full functionality (wiki edits, incremental indexing, MLflow, Airflow)
- **Streamlit Cloud:** stable version for general use

---

## UI Preview  
`<ADD SCREENSHOT OF MAIN UI HERE>`  

---

## Retrieval-Augmented Q&A

- Embeds questions using MiniLM  
- Retrieves relevant chunks using FAISS  
- Optional reranking  
- Answers via GPT-4o (if available) or extractive fallback  
- Includes sentence-to-source attribution and expandable citation previews  

`<ADD SCREENSHOT OF ASK TAB HERE>`

---

## Agent Capabilities

The Agent can:

1. Create new wiki pages  
2. Edit existing pages  
3. Delete pages  
4. Answer normally when no action is intended  

Safeguards:

- All changes require confirmation  
- All edits restricted to `data/processed/wiki/`  
- Harmful content and unsafe titles are blocked  
- Index updates incrementally  

`<ADD SCREENSHOT OF WIKI CREATION HERE>`

---

## Incremental Indexing

The system maintains a vector cache that enables:

- Fast re-embedding of only changed files  
- Efficient deletion of vectors  
- Lightweight rebuilds of the FAISS index  
- Support for prebuilt indexes for instant cold starts

---

## Local vs Cloud

| Feature | Local | Streamlit Cloud |
|--------|-------|------------------|
| Ask (RAG) | ✓ | ✓ |
| Create/Edit/Delete wiki | ✓ | ✓ (safe mode) |
| Upload documents | ✓ | ✓ |
| Incremental indexing | ✓ | ✓ |
| MLflow tracking | ✓ | – |
| Airflow weekly updates | ✓ | – |

---

## MLflow Tracking (Optional, Local)

Tracks retrieval metrics, index versions, evaluation results, and promotion/rollback decisions.

**Promotion logic:**
```txt
If new index improves metrics:
    promote as production
Else:
    rollback to previous version```
	
<ADD SCREENSHOT OF MLFLOW UI HERE>

## Airflow Weekly Pipeline (Optional, Local)

The DAG `airflow/dags/weekly_ingest_and_train.py` automates:

- Fetching new GitLab Handbook pages  
- Rebuilding the index  
- Evaluating retrieval quality  
- Promoting or rolling back based on metrics  

`<ADD SCREENSHOT OF AIRFLOW DAG HERE>`

---

## Quickstart (Local)

```bash
git clone https://github.com/ingo-stallknecht/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent
pip install -r requirements.txt
streamlit run app/streamlit_app.py```

## Optional (development tools)

```bash
pip install -r requirements-dev.txt
bash mlflow/start_mlflow.sh
docker-compose -f docker-compose-airflow.yml up```
