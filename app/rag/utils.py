# app/rag/utils.py

import json
import pathlib

try:
    import yaml  # optional dependency
except ModuleNotFoundError:
    yaml = None


def ensure_dirs():
    """Ensure required data folders exist."""
    for d in ["data/raw", "data/processed/wiki", "data/index", "data/billing"]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def load_cfg(path: str):
    """
    Load configuration from a YAML file if available.

    - If the file does not exist → return {}.
    - If PyYAML is missing → try JSON as a fallback, else return {}.
    - If YAML is available → use yaml.safe_load and return {} on failure.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return {}

    text = p.read_text(encoding="utf-8")

    # If yaml is not available, try JSON as a graceful fallback
    if yaml is None:
        try:
            return json.loads(text)
        except Exception:
            # No YAML and not valid JSON → just use defaults
            return {}

    # Normal path: use YAML
    try:
        cfg = yaml.safe_load(text)
        return cfg or {}
    except Exception:
        # If config is malformed, do not crash the app; just use defaults
        return {}
