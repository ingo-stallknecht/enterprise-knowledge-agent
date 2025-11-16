# scripts/promote_or_rollback.py
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, shutil, time
from pathlib import Path
import mlflow
from app.rag.utils import load_cfg

CFG = load_cfg("configs/settings.yaml")
RET = CFG["retrieval"]
EVAL = CFG.get("eval", {})
TH = EVAL.get("pass_thresholds", {"retrieval_hit_rate": 0.7, "precision_at_k": 0.35, "mrr": 0.45})
EXPERIMENT = CFG.get("mlflow", {}).get("experiment", "EKA_RAG")

INDEX_PATH = Path(RET["faiss_index"])
STORE_PATH = Path(RET["store_json"])
IDX_DIR = Path("data/index")
HISTORY_FILE = IDX_DIR / "history.json"
PTR_FILE = IDX_DIR / "production_paths.json"


def _mlflow_init():
    abs_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(abs_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"[MLflow] tracking_uri={abs_uri}  experiment={EXPERIMENT}")
    return abs_uri


def load_metrics() -> dict:
    p = Path("data/eval/metrics.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_pointer() -> dict:
    return json.loads(PTR_FILE.read_text(encoding="utf-8")) if PTR_FILE.exists() else {}


def write_pointer(index_path: str, store_path: str) -> dict:
    out = {"index_path": index_path, "store_path": store_path, "updated_at": int(time.time())}
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    PTR_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def read_history() -> list:
    return json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []


def write_history(entries: list):
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def create_snapshot() -> Path:
    ver_dir = IDX_DIR / f"prod_snapshot_{int(time.time())}"
    ver_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX_PATH, ver_dir / "handbook.index")
    shutil.copy2(STORE_PATH, ver_dir / "docstore.json")
    return ver_dir


def thresholds_ok(m: dict) -> bool:
    for k, thr in TH.items():
        if float(m.get(k, 0.0)) < float(thr):
            print(f"[promote] FAIL: {k}={m.get(k,0.0):.3f} < {thr:.3f}")
            return False
    return True


def log_common_artifacts():
    for p in [
        "data/eval/metrics.json",
        "data/eval/metrics_detailed.json",
        str(PTR_FILE),
        str(HISTORY_FILE),
    ]:
        if Path(p).exists():
            mlflow.log_artifact(p)


def promote(force: bool = False):
    uri = _mlflow_init()
    metrics = load_metrics()
    with mlflow.start_run(run_name=f"promote-{int(time.time())}"):
        mlflow.set_tag("tracking_uri", uri)
        mlflow.log_param("retrieval.embedder_model", RET.get("embedder_model"))
        mlflow.log_param("retrieval.normalize", RET.get("normalize"))
        mlflow.log_param("retrieval.hybrid_alpha", RET.get("hybrid_alpha"))
        mlflow.log_param("eval.k", EVAL.get("k", RET.get("top_k")))
        mlflow.log_param("thresholds", json.dumps(TH))

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        ok = force or thresholds_ok(metrics)
        mlflow.set_tag("action", "promote")
        mlflow.set_tag("thresholds_ok", str(ok).lower())
        prev_ptr = read_pointer()
        from_ver = prev_ptr.get("index_path", "")

        if ok:
            snap = create_snapshot()
            new_ptr = write_pointer(str(snap / "handbook.index"), str(snap / "docstore.json"))
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


def rollback():
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


def main(mode: str, force: bool = False):
    if mode == "promote":
        promote(force=force)
    elif mode == "rollback":
        rollback()
    else:
        print("Unknown mode. Use --mode promote|rollback")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["promote", "rollback"], required=True)
    ap.add_argument("--force", action="store_true", help="Promote even if thresholds fail")
    args = ap.parse_args()
    main(args.mode, force=args.force)
