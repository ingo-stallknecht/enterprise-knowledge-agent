# app/agent/planner.py
"""
Deterministic agent planner:
plan → rewrite → retrieve → draft → critic
Tidy, human-readable output and clean action payloads.
"""

import os
import re
from typing import Dict, List, Tuple
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.llm.answerer import generate_answer
from app.rag.utils import load_cfg

CFG = load_cfg("configs/settings.yaml")
RET = CFG.get("retrieval", {})
EMB_MODEL = RET.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
NORMALIZE = bool(RET.get("normalize", True))
ALPHA = float(RET.get("hybrid_alpha", 0.6))
RERANK = CFG.get("reranker", {})
RERANK_MODEL = RERANK.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOPN = int(RERANK.get("top_n", 30))

INDEX_PATH = RET.get("faiss_index")
STORE_PATH = RET.get("store_json")

_embedder = Embedder(EMB_MODEL, NORMALIZE)
_reranker = Reranker(RERANK_MODEL)
_index = DocIndex(INDEX_PATH, STORE_PATH)

# ----------------- text utilities -----------------
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"\s+")
_ENDSPACE = re.compile(r"\s+([,.;:!?])")
_MULTI_DOTS = re.compile(r"\.{3,}")

def _strip_md_links(text: str) -> str:
    # converts [label](url) -> label
    return _LINK.sub(r"\1", text or "")

def _smart_preview(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    t = _strip_md_links(text).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    i = max(cut.rfind(" "), cut.rfind("\n"))
    if i > limit // 2:
        cut = cut[:i]
    return cut.rstrip() + "…"

def _tidy_text(text: str) -> str:
    t = (text or "").strip()
    t = _strip_md_links(t)
    t = _ENDSPACE.sub(r"\1", t)          # remove spaces before punctuation
    t = _MULTI_DOTS.sub("…", t)          # normalize long ellipses
    t = _WS.sub(" ", t)                  # collapse whitespace
    return t.strip()

def _basename(path: str) -> str:
    try:
        return os.path.basename(path.replace("\\", "/"))
    except Exception:
        return path

# ----------------- retrieval -----------------
def _retrieve(query: str, k: int = 12) -> Tuple[List[Dict], List[Tuple[Dict, float]]]:
    qv = _embedder.encode([query])
    raw = _index.query_hybrid(qv, query, k=max(k, RERANK_TOPN), alpha=ALPHA)
    candidates = [r for r, _ in raw]
    if candidates:
        pairs = _reranker.rerank(query, candidates, top_k=max(k, 1))
        reranked = [(rec, sc) for rec, sc in pairs]
    else:
        reranked = raw
    top_records = [r for r, _ in reranked[:k]]
    return top_records, reranked

# ----------------- main agent -----------------
def run_agent(message: str, auto_actions: bool = False) -> Dict:
    trace: List[Dict] = []

    # 1) Plan
    plan_steps = ["rewrite_query", "retrieve", "draft", "critic"]
    trace.append({"step": 1, "action": "plan", "output": plan_steps})

    # 2) Rewrite (crisp question)
    query = message.strip()
    if not query.endswith("?"):
        query = query.rstrip(".") + "?"
    query = _tidy_text(query)
    trace.append({"step": 2, "action": "rewrite_query", "output": query})

    # 3) Retrieve (clean, short previews with filenames)
    records, reranked = _retrieve(query, k=12)
    nice_results = []
    for r, sc in reranked[:6]:
        src = r.get("source", "")
        prev = _smart_preview(r.get("text", ""), 180)
        nice_results.append({
            "source": _basename(src),
            "preview": prev
        })
    trace.append({"step": 3, "action": "retrieve", "results": nice_results})

    # 4) Draft answer (tidy)
    answer, _ = generate_answer(records, query, max_chars=900)
    tidy_answer = _tidy_text(answer)
    trace.append({"step": 4, "action": "draft", "output": _smart_preview(tidy_answer, 600)})

    # 5) Critic (human-readable)
    gaps, confidence, actions = _critic(tidy_answer, records)
    pretty_critic = {"confidence": float(confidence)}
    if gaps:     pretty_critic["gaps"] = gaps
    if actions:  pretty_critic["actions"] = actions
    trace.append({"step": 5, "action": "critic", **pretty_critic})

    return {
        "answer": tidy_answer,
        "trace": trace
    }

def _critic(answer: str, records: List[Dict]) -> Tuple[List[str], float, List[Dict]]:
    gaps: List[str] = []
    actions: List[Dict] = []
    conf = 0.70 if len(answer) >= 220 else 0.55

    if "example" not in answer.lower():
        gaps.append("Add concrete, role-specific examples (behavioral or scenario-based).")

    # prepare a clean upsert draft if content seems generic
    needs_wiki = ("hiring" in answer.lower() and "values" in answer.lower())
    if needs_wiki:
        actions.append({
            "type": "upsert_wiki_draft",
            "title": "Values Influence Hiring — Example-Rich Guide",
            "content": (
                "# Values Influence Hiring — Example-Rich Guide\n\n"
                "## Purpose\n"
                "Short, practical reference on weaving CREDIT values into hiring.\n\n"
                "## Examples by Value\n"
                "- **Collaboration** — Behavioral: *Tell me about a time you disagreed with a teammate. How did you reach alignment?*  \n"
                "  Signals: shared ownership, conflict resolution, async artifacts.\n"
                "- **Results** — Evidence: *Walk me through a measurable outcome you improved.*  \n"
                "  Signals: baselines, constraints, shipped delta.\n"
                "- **Efficiency** — Scenario: *Two priorities conflict and time is limited—what do you do first and why?*  \n"
                "  Signals: prioritization heuristics.\n"
                "- **DIB** — Inclusion: *Describe a time you turned feedback into a more inclusive process.*  \n"
                "  Signals: ally behaviors, systemic fixes.\n"
                "- **Iteration** — *Show where you shipped a rough cut and iterated.*  \n"
                "  Signals: MVC mindset.\n"
                "- **Transparency** — *When did you share WIP to unblock others?*  \n"
                "  Signals: written thinking.\n\n"
                "## Anti-Patterns\n"
                "- Rejecting or hiring for vague 'culture fit' — use values alignment with specific evidence instead.\n"
            )
        })

    return gaps, conf, actions
