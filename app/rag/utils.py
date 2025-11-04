# app/rag/utils.py
import os, pathlib, yaml
from mlflow.tracking import MlflowClient


def load_cfg(path: str = "configs/settings.yaml"):
with open(path, "r") as f:
return yaml.safe_load(f)


def ensure_dirs():
for p in [
pathlib.Path("data/raw"),
pathlib.Path("data/processed"),
pathlib.Path("data/processed/wiki"),
pathlib.Path("data/index"),
pathlib.Path("models"),
pathlib.Path("mlflow_artifacts"),
]:
p.mkdir(parents=True, exist_ok=True)


def mlflow_client():
from mlflow import set_tracking_uri
uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlflow_artifacts")
set_tracking_uri(uri)
return MlflowClient()


def get_production_index_paths(model_name: str):
client = mlflow_client()
versions = client.search_model_versions(f"name='{model_name}'")
prod = [v for v in versions if v.current_stage == "Production"]
if not prod:
return None
v = prod[0]
idx = v.tags.get("index_path"); store = v.tags.get("store_path")
return {"index_path": idx, "store_path": store, "version": v.version}