#!/usr/bin/env bash
set -euo pipefail

# Always use the current working directory as the repo root (works reliably on Windows GitBash / PowerShell)
URI="$(python - <<'PY'
import pathlib
root = pathlib.Path.cwd()           # <— repo root if you run this script from the repo root
mlruns = root / "mlruns"
mlruns.mkdir(parents=True, exist_ok=True)
print(mlruns.resolve().as_uri())
PY
)"

echo "---------------------------------------------"
echo " MLflow backend store:"
echo "   $URI"
echo " UI: http://127.0.0.1:5000"
echo "---------------------------------------------"

# Quick peek so you can confirm experiments are visible before opening the UI
python - <<'PY'
from pathlib import Path
root = Path.cwd() / "mlruns"
print("[UI] Using store:", root.resolve())
for p in sorted(root.glob("**/meta.yaml")):
    print("[UI] found:", p)
PY

# Start the UI against THIS exact store
mlflow ui --backend-store-uri "$URI" --host 127.0.0.1 --port 5000
