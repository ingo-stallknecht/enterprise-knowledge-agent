# scripts/promote_or_rollback.py
"""
Promotion/rollback based on eval metrics and thresholds in configs/settings.yaml.
- On promotion: copies current index/store into a timestamped snapshot dir and
  writes data/index/production_paths.json to point to it. Logs to MLflow.
- On rollback: switches pointer to the previous snapshot (if any). Logs to MLflow.

Artifacts logged to MLflow:
- data/eval/metrics.json
- promoted snapshot files (handbook.index, docstore.json)
"""

import argparse, json, pathlib, shutil, time, re
import mlflow
from app.rag.utils import load_cfg

CFG = load_cfg("configs/settings.yaml")
TH = CFG.get("eval", {}).get("pass_thresholds", {"retrieval_hit_rate": 0.7})
RET = CFG["retrieval"]

PROD_PTR = pathlib.Path("data/index/production_paths.json")
SNAP_DIR = pathlib.Path("data/index")

def _snapshots():
    patt = re.compile(r"^prod_snapshot_(\d+)$")
    snaps = []
    if SNAP_DIR.exists():
        for d in SNAP_DIR.iterdir():
            if d.is_dir():
                m = patt.match(d.name)
                if m:
                    snaps.append((int(m.group(1)), d))
    snaps.sort(key=lambda x: x[0])  # by timestamp asc
    return snaps  # list of (ts, Path)

def load_metrics():
    p = pathlib.Path("data/eval/metrics.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_production_pointer(index_path: str, store_path: str):
    out = {"index_path": index_path, "store_path": store_path, "updated_at": int(time.time())}
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    PROD_PTR.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

def current_pointer():
    if not PROD_PTR.exists():
        return {}
    try:
        return json.loads(PROD_PTR.read_text(encoding="utf-8"))
    except Exception:
        return {}

def promote():
    # copy current index to a versioned folder (for audit)
    ts = int(time.time())
    ver_dir = SNAP_DIR / f"prod_snapshot_{ts}"
    ver_dir.mkdir(parents=True, exist_ok=True)
    src_idx = pathlib.Path(RET["faiss_index"])
    src_store = pathlib.Path(RET["store_json"])
    if not src_idx.exists() or not src_store.exists():
        raise FileNotFoundError("Index or store missing. Run build_index + eval first.")
    dst_idx = ver_dir / "handbook.index"
    dst_store = ver_dir / "docstore.json"
    shutil.copy2(src_idx, dst_idx)
    shutil.copy2(src_store, dst_store)

    # update pointer
    ptr = save_production_pointer(str(dst_idx), str(dst_store))
    print(f"[promote] success → {ver_dir}")

    # log artifacts
    try:
        mlflow.log_artifact("data/eval/metrics.json")
    except Exception:
        pass
    try:
        mlflow.log_artifact(str(dst_idx))
        mlflow.log_artifact(str(dst_store))
    except Exception:
        pass
    mlflow.set_tag("promotion", "success")
    mlflow.set_tag("snapshot_dir", ver_dir.name)
    mlflow.set_tag("pointer_updated_at", str(ptr["updated_at"]))

def rollback():
    snaps = _snapshots()
    if not snaps:
        print("[rollback] no snapshots found")
        mlflow.set_tag("rollback", "no_snapshots")
        return

    # Determine current snapshot (if pointer matches one), then pick the previous
    ptr = current_pointer()
    cur_snap = None
    for ts, d in snaps[::-1]:  # newest first
        if ptr.get("index_path", "").startswith(str(d)) or ptr.get("store_path", "").startswith(str(d)):
            cur_snap = (ts, d)
            break

    if cur_snap:
        # choose the snapshot immediately older than current
        indices = [ts for ts, _ in snaps]
        i = indices.index(cur_snap[0])
        target = snaps[i - 1] if i - 1 >= 0 else None
    else:
        # no pointer or not matching → roll back to the newest snapshot
        target = snaps[-1]

    if not target:
        print("[rollback] no older snapshot to roll back to")
        mlflow.set_tag("rollback", "no_older_snapshot")
        return

    ts, d = target
    idx = d / "handbook.index"
    store = d / "docstore.json"
    if not idx.exists() or not store.exists():
        print(f"[rollback] snapshot incomplete: {d}")
        mlflow.set_tag("rollback", "snapshot_incomplete")
        return

    ptr = save_production_pointer(str(idx), str(store))
    print(f"[rollback] pointer → {d}")
    mlflow.set_tag("rollback", "success")
    mlflow.set_tag("snapshot_dir", d.name)
    mlflow.set_tag("pointer_updated_at", str(ptr["updated_at"]))

def main(mode: str):
    metrics = load_metrics()
    mlflow.set_tracking_uri(CFG.get("mlflow", {}).get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(CFG.get("mlflow", {}).get("experiment", "EKA_RAG"))

    with mlflow.start_run(run_name=f"{mode}-{int(time.time())}"):
        # log metrics if present
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        if mode == "promote":
            # check thresholds
            ok = True
            for k, thr in TH.items():
                if float(metrics.get(k, 0.0)) < float(thr):
                    ok = False
                    print(f"[promote] {k} {float(metrics.get(k, 0.0)):.3f} < threshold {thr:.3f} → skip promotion")
            if ok:
                promote()
        elif mode == "rollback":
            rollback()
        else:
            print("Unknown mode. Use --mode promote|rollback")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["promote", "rollback"], required=True)
    args = ap.parse_args()
    main(args.mode)
