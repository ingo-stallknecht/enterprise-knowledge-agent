# scripts/promote_or_rollback.py
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.utils import load_cfg  # noqa: E402

CFG = load_cfg("configs/settings.yaml")
RET = CFG["retrieval"]
EVAL = CFG.get("eval", {})
TH = EVAL.get(
    "pass_thresholds",
    {
        "retrieval_hit_rate": 0.7,
        "precision_at_k": 0.35,
        "mrr": 0.45,
    },
)
EXPERIMENT = CFG.get("mlflow", {}).get("experiment", "EKA_RAG")

INDEX_PATH = Path(RET["faiss_index"])
STORE_PATH = Path(RET["store_json"])
IDX_DIR = Path("data/index")
HISTORY_FILE = IDX_DIR / "history.json"
PTR_FILE = IDX_DIR / "production_paths.json"


def _mlflow_init() -> str:
    """Configure MLflow to use a local file backend and return the tracking URI."""
    abs_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(abs_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"[MLflow] tracking_uri={abs_uri}  experiment={EXPERIMENT}")
    return abs_uri


def load_metrics() -> Dict[str, Any]:
    """Load the latest evaluation metrics from data/eval/metrics.json."""
    p = Path("data/eval/metrics.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_pointer() -> Dict[str, Any]:
    """Read the current production pointer (index/store paths)."""
    if not PTR_FILE.exists():
        return {}
    return json.loads(PTR_FILE.read_text(encoding="utf-8"))


def write_pointer(index_path: str, store_path: str) -> Dict[str, Any]:
    """Write the production pointer to disk and return it."""
    out = {
        "index_path": index_path,
        "store_path": store_path,
        "updated_at": int(time.time()),
    }
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    PTR_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def read_history() -> List[Dict[str, Any]]:
    """Read promotion history entries from disk."""
    if not HISTORY_FILE.exists():
        return []
    return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))


def write_history(entries: List[Dict[str, Any]]) -> None:
    """Persist the full promotion history."""
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def create_snapshot() -> Path:
    """
    Create a snapshot directory with the current index + docstore.

    Returns the snapshot directory path.
    """
    ver_dir = IDX_DIR / f"prod_snapshot_{int(time.time())}"
    ver_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX_PATH, ver_dir / "handbook.index")
    shutil.copy2(STORE_PATH, ver_dir / "docstore.json")
    return ver_dir


def thresholds_ok(metrics: Dict[str, Any]) -> bool:
    """Check if evaluation metrics meet the configured thresholds."""
    for key, thr in TH.items():
        val = float(metrics.get(key, 0.0))
        if val < float(thr):
            print(f"[promote] FAIL: {key}={val:.3f} < {thr:.3f}")
            return False
    return True


def log_common_artifacts() -> None:
    """Log key JSON files to the current MLflow run."""
    artifacts = [
        "data/eval/metrics.json",
        "data/eval/metrics_detailed.json",
        str(PTR_FILE),
        str(HISTORY_FILE),
    ]
    for path_str in artifacts:
        path = Path(path_str)
        if path.exists():
            mlflow.log_artifact(str(path))


def promote(force: bool = False) -> None:
    """
    Promote the current index/docstore to 'production' if thresholds pass.

    When successful:
    - Create a new snapshot under data/index/prod_snapshot_*
    - Update production_paths.json pointer
    - Append to history.json

    When thresholds fail (and not forced), do nothing.
    """
    uri = _mlflow_init()
    metrics = load_metrics()

    with mlflow.start_run(run_name=f"promote-{int(time.time())}"):
        mlflow.set_tag("tracking_uri", uri)
        mlflow.log_param("retrieval.embedder_model", RET.get("embedder_model"))
        mlflow.log_param("retrieval.normalize", RET.get("normalize"))
        mlflow.log_param("retrieval.hybrid_alpha", RET.get("hybrid_alpha"))
        mlflow.log_param("eval.k", EVAL.get("k", RET.get("top_k")))
        mlflow.log_param("thresholds", json.dumps(TH))

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        ok = force or thresholds_ok(metrics)
        mlflow.set_tag("action", "promote")
        mlflow.set_tag("thresholds_ok", str(ok).lower())

        prev_ptr = read_pointer()
        from_ver = prev_ptr.get("index_path", "")

        if ok:
            snap = create_snapshot()
            new_ptr = write_pointer(
                str(snap / "handbook.index"),
                str(snap / "docstore.json"),
            )
            hist = read_history()
            hist.append(
                {
                    "snapshot_dir": str(snap),
                    "index_path": new_ptr["index_path"],
                    "store_path": new_ptr["store_path"],
                    "ts": new_ptr["updated_at"],
                }
            )
            write_history(hist)

            mlflow.set_tag("promotion", "success")
            mlflow.set_tag("from_version", from_ver)
            mlflow.set_tag("to_version", new_ptr["index_path"])
            print(f"[promote] success → {snap}")
        else:
            mlflow.set_tag("promotion", "skipped")
            mlflow.set_tag("from_version", from_ver)
            mlflow.set_tag("to_version", from_ver)
            print("[promote] thresholds not met → skipped")

        log_common_artifacts()


def rollback() -> None:
    """
    Roll back the production pointer to the previous snapshot (if any).

    This does NOT delete any snapshot; it only moves the production pointer.
    """
    uri = _mlflow_init()

    with mlflow.start_run(run_name=f"rollback-{int(time.time())}"):
        mlflow.set_tag("tracking_uri", uri)
        mlflow.set_tag("action", "rollback")

        hist = read_history()
        if not hist:
            mlflow.set_tag("rollback", "no_history")
            print("[rollback] no history → noop")
            return

        cur = read_pointer()
        cur_idx = None
        for i, h in enumerate(hist):
            if h.get("index_path") == cur.get("index_path") and h.get("store_path") == cur.get(
                "store_path"
            ):
                cur_idx = i
                break

        target_idx = (len(hist) - 2) if cur_idx is None else (cur_idx - 1)
        if target_idx < 0:
            mlflow.set_tag("rollback", "no_previous")
            print("[rollback] already at oldest snapshot → noop")
            log_common_artifacts()
            return

        target = hist[target_idx]
        new_ptr = write_pointer(target["index_path"], target["store_path"])
        mlflow.set_tag("rollback", "success")
        mlflow.set_tag("from_version", cur.get("index_path", ""))
        mlflow.set_tag("to_version", new_ptr["index_path"])
        print(f"[rollback] pointer → {new_ptr['index_path']}")
        log_common_artifacts()


def main(mode: str, force: bool = False) -> None:
    if mode == "promote":
        promote(force=force)
    elif mode == "rollback":
        rollback()
    else:
        print("Unknown mode. Use --mode promote|rollback")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["promote", "rollback"],
        required=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promote even if thresholds fail",
    )
    args = parser.parse_args()
    main(args.mode, force=args.force)
