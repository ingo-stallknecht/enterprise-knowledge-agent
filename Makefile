.PHONY: init api airflow mlflow pipeline index ingest eval promote rollback test fmt


init:
python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt


api:
. .venv/bin/activate && uvicorn app.server:app --reload --port 8000


airflow:
docker compose up airflow-init -d; docker compose up airflow-webserver airflow-scheduler -d


mlflow:
bash mlflow/start_mlflow.sh


ingest:
. .venv/bin/activate && python scripts/fetch_gitlab_handbook.py


index:
. .venv/bin/activate && python scripts/build_index.py


eval:
. .venv/bin/activate && python scripts/eval_rag.py


promote:
. .venv/bin/activate && python scripts/promote_or_rollback.py --mode promote


rollback:
. .venv/bin/activate && python scripts/promote_or_rollback.py --mode rollback


pipeline: ingest index eval promote


test:
. .venv/bin/activate && pytest -q


fmt:
. .venv/bin/activate && python -m nltk.downloader punkt