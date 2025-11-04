# scripts/promote_or_rollback.py
import sys, os, argparse, json, pathlib, yaml
import mlflow
from mlflow.tracking import MlflowClient

# Ensure repo root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_latest_run_metrics(metrics_path="data/eval/metrics.json") -> dict:
    """Load latest evaluation metrics from disk."""
    if not os.path.exists(metrics_path):
        print("[promote] No metrics file found.")
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)

def promote_model(client: MlflowClient, model_name: str):
    """Promote the latest model version to Production."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("[promote] No model versions found.")
        return
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(model_name, latest.version, stage="Production")
    print(f"[promote] Promoted version {latest.version} → Production")

def rollback_model(client: MlflowClient, model_name: str):
    """Rollback to the previous Production model."""
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if len(prod_versions) <= 1:
        print("[rollback] No previous production version found.")
        return
    # keep the second-last production model
    sorted_versions = sorted(prod_versions, key=lambda v: int(v.version))
    current = sorted_versions[-1]
    previous = sorted_versions[-2]
    client.transition_model_version_stage(model_name, current.version, stage="Archived")
    client.transition_model_version_stage(model_name, previous.version, stage="Production")
    print(f"[rollback] Rolled back to version {previous.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["promote", "rollback"], required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/settings.yaml", encoding="utf-8"))
    ml_cfg = cfg.get("mlflow", {})

    tracking_uri = ml_cfg.get("tracking_uri", "file:./mlflow_artifacts")
    model_name = ml_cfg.get("registered_model", "EKA_RAG_Index")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Ensure model exists in registry
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    metrics = get_latest_run_metrics()
    thresholds = cfg.get("eval", {}).get("pass_thresholds", {})

    if args.mode == "promote":
        recall = metrics.get("recall_at_k", 0)
        recall_thr = thresholds.get("recall_at_k", 0.0)
        if recall >= recall_thr:
            promote_model(client, model_name)
        else:
            print(f"[promote] recall {recall:.3f} < threshold {recall_thr:.3f} → skip promotion")
    elif args.mode == "rollback":
        rollback_model(client, model_name)
