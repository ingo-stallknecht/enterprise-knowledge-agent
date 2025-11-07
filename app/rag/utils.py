# app/rag/utils.py
import json, pathlib, yaml

def ensure_dirs():
    for p in [
        "data/raw",
        "data/processed",
        "data/processed/wiki",
        "data/index",
        "data/eval",
        "data/billing",
    ]:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_production_index_paths(model_name: str):
    """
    Local pointer written into data/index/production_paths.json
    Returns {"index_path": "...", "store_path": "..."} or {}.
    """
    ptr = pathlib.Path("data/index/production_paths.json")
    if ptr.exists():
        try:
            return json.loads(ptr.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
