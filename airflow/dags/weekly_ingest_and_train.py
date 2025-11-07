from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# ---- Configure where your repo is mounted inside the Airflow runtime ----
# If you use Docker, mount the repo at /repo (recommended). Otherwise set this to the absolute path on the host.
REPO = os.environ.get("EKA_REPO", "/repo")

# Optional: if you keep a venv inside the repo (local dev),
# we’ll prefer its python; otherwise fall back to system python in the container.
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

# Weekly Monday 06:00
with DAG(
    dag_id="eka_weekly_ingest_and_train",
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * 1",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["eka", "rag"],
) as dag:

    # 1) Fetch curated handbook pages
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"cd {REPO} && {PY} scripts/fetch_gitlab_handbook.py",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 2) Build FAISS index
    index = BashOperator(
        task_id="index",
        bash_command=f"cd {REPO} && {PY} -m scripts.build_index",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 3) Evaluate retrieval
    eval_rag = BashOperator(
        task_id="eval",
        bash_command=f"cd {REPO} && {PY} scripts/eval_rag.py",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    # 4) Promote if thresholds pass (logs to MLflow, snapshots prod, updates pointer)
    promote = BashOperator(
        task_id="promote",
        bash_command=f"cd {REPO} && {PY} scripts/promote_or_rollback.py --mode promote",
        env={"PYTHONPATH": REPO, **os.environ},
    )

    ingest >> index >> eval_rag >> promote
