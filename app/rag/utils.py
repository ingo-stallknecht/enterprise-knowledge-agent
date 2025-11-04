# app/rag/utils.py
import os
import yaml
import pathlib
from typing import Dict

def ensure_dirs():
    """Ensure key data directories exist."""
    for d in ["data/raw", "data/processed", "data/index", "mlflow_artifacts"]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def load_cfg(path: str) -> dict:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_production_index_paths(model_name: str) -> Dict[str, str]:
    """
    Check MLflow registry for a Production model version and return its artifact paths.
    Works gracefully even if MLflow or the registry is not available.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        prod = [v for v in versions if v.current_stage == "Production"]
        if not prod:
            return {}
        latest = sorted(prod, key=lambda v: int(v.version))[-1]
        run_id = latest.run_id
        artifact_uri = client.get_run(run_id).info.artifact_uri
        base = artifact_uri.replace("file://", "")
        return {
            "index_path": os.path.join(base, "handbook.index"),
            "store_path": os.path.join(base, "docstore.json"),
        }
    except Exception:
        return {}
