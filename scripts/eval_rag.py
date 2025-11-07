# scripts/eval_rag.py
"""
Lightweight retrieval eval (hybrid):
- retrieval_hit_rate, mrr_at_k
Writes data/eval/metrics.json
"""
import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
warnings.filterwarnings(
    "ignore",
    message=r"`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
    module=r".*transformers.*",
)

import json
from typing import List, Dict

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **_: x

from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import load_cfg

CFG = load_cfg("configs/settings.yaml")
RET = CFG["retrieval"]
K = int(RET.get("top_k", 12))
ALPHA = float(RET.get("hybrid_alpha", 0.6))
SHOW_PROGRESS = bool(RET.get("progress", False) or os.getenv("EKA_PROGRESS", "0") in ("1", "true", "True"))

EVAL_SET: List[Dict[str, str]] = [
    {"q": "company values", "kw": "values"},
    {"q": "communication guidelines", "kw": "communication"},
    {"q": "product management", "kw": "product"},
    {"q": "engineering practices", "kw": "engineering"},
]

def first_rank_with_keyword(results: List[Dict], keyword: str) -> int:
    kw = (keyword or "").lower().strip()
    if not kw:
        return -1
    for i, rec in enumerate(results):
        txt = (rec.get("text") or "").lower()
        if kw in txt:
            return i
    return -1

def main():
    os.makedirs("data/eval", exist_ok=True)

    embedder = Embedder(RET["embedder_model"], RET["normalize"], progress=SHOW_PROGRESS)
    index = DocIndex(RET["faiss_index"], RET["store_json"])

    if index.index is None or not index.store:
        print("[eval] No index/store found. Build the index first (scripts/build_index.py).")
        raise SystemExit(1)

    hits, rr_sum = 0, 0.0
    per_query = []

    for row in tqdm(EVAL_SET, desc="Evaluating", disable=not SHOW_PROGRESS):
        q = row["q"]; kw = row["kw"]
        qv = embedder.encode([q])

        hybrid = index.query_hybrid(qv, q, k=K, alpha=ALPHA)
        top_records = [r for r, _ in hybrid[:K]]

        concat = " ".join((r.get("text") or "") for r in top_records).lower()
        hit = int(kw.lower() in concat); hits += hit

        rank = first_rank_with_keyword(top_records, kw)
        rr = 1.0 / (rank + 1) if rank >= 0 else 0.0; rr_sum += rr

        per_query.append({
            "query": q, "keyword": kw, "hit": bool(hit),
            "rr": rr, "first_rank": (rank + 1) if rank >= 0 else None
        })

    total = len(EVAL_SET)
    metrics = {
        "eval_queries": total, "k": K, "mode": "hybrid", "alpha": ALPHA,
        "retrieval_hit_rate": hits / max(1, total),
        "mrr_at_k": rr_sum / max(1, total),
        "per_query": per_query,
    }

    with open("data/eval/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[eval] metrics written to data/eval/metrics.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_query"}, indent=2))

if __name__ == "__main__":
    main()

