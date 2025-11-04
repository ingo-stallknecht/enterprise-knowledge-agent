# Enterprise Knowledge Agent — Cross-platform Makefile (robust venv detection)

.SILENT:
.ONESHELL:
SHELL := /usr/bin/sh

.PHONY: init api airflow mlflow pipeline index ingest eval promote rollback test fmt test-api ui print-which

# Resolve venv executables (Windows Git Bash vs Unix)
PYTHON := $(shell if [ -f .venv/Scripts/python.exe ]; then printf ".venv/Scripts/python.exe"; \
                   elif [ -f .venv/bin/python ]; then printf ".venv/bin/python"; \
                   else printf "python"; fi)
STREAMLIT := $(shell if [ -f .venv/Scripts/streamlit.exe ]; then printf ".venv/Scripts/streamlit.exe"; \
                      elif [ -f .venv/bin/streamlit ]; then printf ".venv/bin/streamlit"; \
                      else printf "streamlit"; fi)

print-which:
	@echo "PYTHON=$(PYTHON)"
	@echo "STREAMLIT=$(STREAMLIT)"

init:
	python -m venv .venv && \
	python -m pip install --upgrade pip && \
	python -m pip install -r requirements.txt

api:
	"$(PYTHON)" -m uvicorn app.server:app --reload --port 8000

airflow:
	docker compose up airflow-init -d && \
	docker compose up airflow-webserver airflow-scheduler -d

mlflow:
	bash mlflow/start_mlflow.sh

ingest:
	"$(PYTHON)" scripts/fetch_gitlab_handbook.py

index:
	"$(PYTHON)" -m scripts.build_index

eval:
	"$(PYTHON)" scripts/eval_rag.py

promote:
	"$(PYTHON)" scripts/promote_or_rollback.py --mode promote

rollback:
	"$(PYTHON)" scripts/promote_or_rollback.py --mode rollback

pipeline: ingest index eval promote

test:
	"$(PYTHON)" -m pytest -q

fmt:
	"$(PYTHON)" -m nltk.downloader punkt

test-api:
	"$(PYTHON)" scripts/test_api_local.py

ui:
	"$(STREAMLIT)" run app/streamlit_app.py
