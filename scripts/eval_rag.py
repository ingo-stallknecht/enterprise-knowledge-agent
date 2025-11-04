# scripts/eval_rag.py
import sys, os, json, pathlib, yaml, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.embedder import Embedder
from app.rag.index import DocIndex

def contains_any(text: str, gold_tokens: list[str]) -> bool:
    t = text.lower()
    return any(g.lower() in t for g in gold_tokens if g)

def eval_questions(questions, gold_lists, embedder, index: DocIndex, k: int):
    """Return recall_at_k and answer_contains_gold_frac using chunk-text heuristics."""
    # encode questions
    q_vecs = embedder.encode(questions)

    hits = 0
    contains = 0
    total = len(questions)

    for qv, gold in zip(q_vecs, gold_lists):
        results = index.query(qv.reshape(1, -1), k=k)  # list[(record, score)]
        hay = "\n".join([r[0]["text"] for r in results])
        ok = contains_any(hay, gold if isinstance(gold, list) else [gold])
        if ok:
            hits += 1
            contains += 1

    recall_at_k = hits / total if total else 0.0
    answer_contains_gold_frac = contains / total if total else 0.0
    return {
        "recall_at_k": recall_at_k,
        "answer_contains_gold_frac": answer_contains_gold_frac,
        "n_queries": total,
        "k": k,
    }

# Load config
cfg = yaml.safe_load(open("configs/settings.yaml", encoding="utf-8"))
eval_cfg = cfg.get("eval", {})
retrieval_cfg = cfg.get("retrieval", {})

q_path = pathlib.Path(eval_cfg.get("questions_path", "configs/eval_questions.jsonl"))
if not q_path.exists():
    demo = [
        {"question": "What are GitLab's values?", "gold": ["values", "collaboration"]},
        {"question": "Where is the engineering handbook described?", "gold": ["engineering"]},
        {"question": "How do I propose a change to the handbook?", "gold": ["merge request", "MR"]},
    ]
    with open(q_path, "w", encoding="utf-8") as f:
        for d in demo:
            f.write(json.dumps(d) + "\n")

# Read questions
questions, golds = [], []
for line in open(q_path, encoding="utf-8"):
    obj = json.loads(line)
    questions.append(obj["question"])
    golds.append(obj.get("gold", []))

# Components
embedder = Embedder(
    retrieval_cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"),
    retrieval_cfg.get("normalize", True),
)
index = DocIndex(
    retrieval_cfg.get("faiss_index", "data/index/handbook.index"),
    retrieval_cfg.get("store_json", "data/index/docstore.json"),
)

k = int(eval_cfg.get("k", 6))
metrics = eval_questions(questions, golds, embedder, index, k)

# Save + print
out_dir = pathlib.Path("data/eval"); out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "metrics.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

for mk, mv in metrics.items():
    print(f"{mk}: {mv}")

print(f"[eval] metrics written to {out_path}")
