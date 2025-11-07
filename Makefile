# Local-only Makefile (with MLflow/Airflow hooks)

.SILENT:
.ONESHELL:
SHELL := /usr/bin/sh

.PHONY: init api ui airflow mlflow pipeline index ingest eval promote rollback test fmt print-which

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
	"$(PYTHON)" -m uvicorn app.server:app --port 8000 --log-level info

ui:
	"$(STREAMLIT)" run app/streamlit_app.py

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
	"$(PYTHON)" -m scripts.eval_rag

promote:
	"$(PYTHON)" -m scripts.promote_or_rollback --mode promote

rollback:
	"$(PYTHON)" -m scripts.promote_or_rollback --mode rollback

pipeline: ingest index eval promote

test:
	"$(PYTHON)" -m pytest -q

fmt:
	"$(PYTHON)" -m nltk.downloader punkt
