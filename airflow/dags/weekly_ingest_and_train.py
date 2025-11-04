from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import yaml, os


with DAG(
dag_id="eka_weekly_ingest_and_train",
start_date=datetime(2024,1,1),
schedule_interval="0 6 * * 1", # Monday 06:00
catchup=False,
tags=["eka","rag"],
) as dag:


ingest = BashOperator(
task_id="ingest",
bash_command=". /repo/.venv/bin/activate && python /repo/scripts/fetch_gitlab_handbook.py",
)


index = BashOperator(
task_id="index",
bash_command=". /repo/.venv/bin/activate && python /repo/scripts/build_index.py",
)


eval = BashOperator(
task_id="eval",
bash_command=". /repo/.venv/bin/activate && python /repo/scripts/eval_rag.py",
)


promote = BashOperator(
task_id="promote",
bash_command=". /repo/.venv/bin/activate && python /repo/scripts/promote_or_rollback.py --mode promote || true",
)


ingest >> index >> eval >> promote