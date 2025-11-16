# scripts/copy_index_to_prebuilt.py
"""
Copy the currently built FAISS index + docstore into data/index/prebuilt/
so Streamlit Cloud can use them as a prebuilt index (fast cold starts).

Usage (from repo root):
    python scripts/copy_index_to_prebuilt.py
"""

import pathlib
import shutil
import sys

# ----------------------------------------------------------------------
# Make sure the repo root is on sys.path so `app.*` imports work
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.utils import load_cfg, ensure_dirs  # noqa: E402


def main() -> None:
    cfg_path = ROOT / "configs" / "settings.yaml"

    CFG = load_cfg(str(cfg_path)) if cfg_path.exists() else {}
    RET = CFG.get("retrieval", {})

    index_path = ROOT / RET.get("faiss_index", "data/index/handbook.index")
    store_path = ROOT / RET.get("store_json", "data/index/docstore.json")
    prebuilt_dir = ROOT / "data" / "index" / "prebuilt"

    ensure_dirs()
    prebuilt_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists() or not store_path.exists():
        raise SystemExit(
            f"Index not found.\n"
            f"Expected:\n  {index_path}\n  {store_path}\n\n"
            "Run the Streamlit app locally once (so it fetches + embeds) "
            "and then re-run this script."
        )

    dst_index = prebuilt_dir / index_path.name
    dst_store = prebuilt_dir / store_path.name

    shutil.copy2(index_path, dst_index)
    shutil.copy2(store_path, dst_store)

    print("Copied index to prebuilt:")
    print(f"  {index_path} -> {dst_index}")
    print(f"  {store_path} -> {dst_store}")


if __name__ == "__main__":
    main()
