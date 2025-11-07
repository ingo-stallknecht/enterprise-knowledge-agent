# scripts/eval_rag.py
"""
Evaluates retrieval (hit rate, precision@k, MRR) and logs to MLflow with a local file backend.
"""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import json, time, yaml
from typing import Dict, List, Tuple
from pathlib import Path
import mlflow

from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import load_cfg

CFG = load_cfg("configs/settings.yaml")
RET = CFG["retrieval"]
EVAL_CFG = CFG.get("eval", {})
K = int(EVAL_CFG.get("k", RET.get("top_k", 12)))
QUESTIONS_FILE = EVAL_CFG.get("questions_file", "configs/eval_questions.yaml")
EXPERIMENT = CFG.get("mlflow", {}).get("experiment", "EKA_RAG")

def load_questions(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return [{"q": "How are values applied in performance reviews?", "must_terms": ["values","performance"], "relevant_sources": ["values.md"]}]
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []

def normalize_src(src: str) -> str: return (src or "").replace("\\", "/")

def text_contains_any(t: str, terms: List[str]) -> bool:
    t = (t or "").lower()
    return any(w.lower() in t for w in (terms or []))

def source_matches(src: str, keys: List[str]) -> bool:
    s = normalize_src(src).lower()
    return any(k.lower() in s for k in (keys or []))

def compute_metrics_for_query(res_topk: List[Tuple[Dict, float]], must_terms: List[str], relevant_sources: List[str]) -> Dict:
    relevant_flags, rr, first_rel_rank = [], 0.0, None
    for i, (rec, _sc) in enumerate(res_topk, start=1):
        txt, src = rec.get("text") or "", rec.get("source") or ""
        is_rel = (must_terms and text_contains_any(txt, must_terms)) or (relevant_sources and source_matches(src, relevant_sources))
        relevant_flags.append(bool(is_rel))
        if is_rel and first_rel_rank is None:
            first_rel_rank, rr = i, 1.0 / i
    hits = 1.0 if any(relevant_flags) else 0.0
    prec = float(sum(relevant_flags)) / float(len(relevant_flags) or 1)
    return {"hit": hits, "precision_at_k": prec, "rr": rr, "first_rel_rank": first_rel_rank or 0, "relevant_flags": relevant_flags}

def main():
    embedder = Embedder(RET["embedder_model"], RET["normalize"])
    index = DocIndex(RET["faiss_index"], RET["store_json"])
    questions = load_questions(QUESTIONS_FILE)

    results = []
    for q in questions:
        qtext = (q.get("q") or "").strip()
        if not qtext: continue
        qv = embedder.encode([qtext])
        pairs = index.query_hybrid(qv, qtext, k=K, alpha=RET.get("hybrid_alpha", 0.6))
        metrics = compute_metrics_for_query(pairs[:K], q.get("must_terms", []), q.get("relevant_sources", []))
        results.append({"q": qtext, "must_terms": q.get("must_terms", []), "relevant_sources": q.get("relevant_sources", []),
                        "metrics": metrics, "topk_sources": [normalize_src(r.get("source","")) for r,_ in pairs[:K]]})

    n = len(results) or 1
    retrieval_hit_rate = sum(r["metrics"]["hit"] for r in results) / n
    precision_at_k     = sum(r["metrics"]["precision_at_k"] for r in results) / n
    mrr                = sum(r["metrics"]["rr"] for r in results) / n
    metrics = {"num_questions": n, "k": K, "retrieval_hit_rate": retrieval_hit_rate, "precision_at_k": precision_at_k, "mrr": mrr, "timestamp": int(time.time())}

    out_dir = Path("data/eval"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics_detailed.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("[eval] metrics written to data/eval/metrics.json")
    print(metrics)

    # === Unified MLflow backend ===
    abs_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(abs_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"[MLflow] tracking_uri={abs_uri}  experiment={EXPERIMENT}")

    with mlflow.start_run(run_name=f"eval-{int(time.time())}"):
        # Make the URI easy to see inside the UI
        mlflow.set_tag("tracking_uri", abs_uri)
        mlflow.log_param("retrieval.embedder_model", RET.get("embedder_model"))
        mlflow.log_param("retrieval.normalize", RET.get("normalize"))
        mlflow.log_param("retrieval.hybrid_alpha", RET.get("hybrid_alpha"))
        mlflow.log_param("eval.k", K)
        mlflow.log_param("eval.questions_file", QUESTIONS_FILE)
        for k, v in metrics.items():
            if isinstance(v, (int, float)): mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        mlflow.log_artifact(str(out_dir / "metrics_detailed.json"))

if __name__ == "__main__":
    main()
