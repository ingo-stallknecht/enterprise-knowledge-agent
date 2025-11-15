# airflow/dags/weekly_ingest_and_train.py
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

"""
Weekly ingestion and training pipeline for the Enterprise Knowledge Agent.

This DAG is intended for local development only. It assumes that:

- The git repository is mounted inside the Airflow runtime and the path
  is available as the environment variable EKA_REPO (default: /repo).
- The repository contains the following scripts:
    scripts/fetch_gitlab_handbook.py
    scripts/build_index.py            or a module scripts/build_index
    scripts/eval_rag.py
    scripts/promote_or_rollback.py

The high level flow is:

1. ingest      - fetch curated GitLab Handbook pages and write markdown files
2. index       - build or refresh the FAISS index and docstore
3. eval        - evaluate retrieval quality and write metrics
4. promote     - promote the new index if thresholds pass, rollback otherwise

Nothing from Airflow itself (logs, metastore, etc.) needs to be pushed to git.
Only this DAG file and the scripts it calls belong in version control.
"""

# Where the repo is mounted inside the Airflow runtime.
# If you use Docker, mount the repo to /repo and set EKA_REPO=/repo.
REPO = os.environ.get("EKA_REPO", "/repo")

# Optional: prefer the repo's virtual environment python if it exists.
# Otherwise, fall back to "python" from the Airflow container.
VENV_PY = os.path.join(REPO, ".venv", "bin", "python")
PY = VENV_PY if os.path.exists(VENV_PY) else "python"

DEFAULT_ARGS = {
    "owner": "eka",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Weekly Monday at 06:00
with DAG(
    dag_id="eka_weekly_ingest_and_train",
    description="Weekly ingest, index, evaluate and promote pipeline for EKA.",
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * 1",
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["eka", "rag", "local-dev"],
) as dag:

    # 1) Fetch curated handbook pages (GitLab Handbook subset)
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"cd {REPO} && {PY} scripts/fetch_gitlab_handbook.py",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 2) Build FAISS index for the current corpus
    index = BashOperator(
        task_id="index",
        bash_command=f"cd {REPO} && {PY} scripts/build_index.py",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 3) Evaluate retrieval and log metrics
    eval_rag = BashOperator(
        task_id="eval",
        bash_command=f"cd {REPO} && {PY} scripts/eval_rag.py",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 4) Promote or rollback based on evaluation thresholds
    promote = BashOperator(
        task_id="promote",
        bash_command=(
            f"cd {REPO} && {PY} scripts/promote_or_rollback.py --mode promote"
        ),
        env={"PYTHONPATH": REPO, **os.environ},
    )

    ingest >> index >> eval_rag >> promote
