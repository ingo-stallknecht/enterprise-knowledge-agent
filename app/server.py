# app/server.py
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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

import sys, pathlib, sqlite3, time, threading, traceback
from collections import deque
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slugify import slugify
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import ensure_dirs, load_cfg, get_production_index_paths
from app.llm.answerer import generate_answer
from app.rag.reranker import Reranker
from app.agent.planner import run_agent
from scripts.build_index import build_index as build_index_fn

# ---------------------------
# Setup & config
# ---------------------------
ensure_dirs()
CFG = load_cfg("configs/settings.yaml")

RET = CFG.get("retrieval", {})
CFG_INDEX_PATH = RET.get("faiss_index", "data/index/handbook.index")
CFG_STORE_PATH = RET.get("store_json", "data/index/docstore.json")
EMB_MODEL = RET.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
NORMALIZE = bool(RET.get("normalize", True))
TOP_K_DEFAULT = int(RET.get("top_k", 12))
ALPHA = float(RET.get("hybrid_alpha", 0.6))

RERANK = CFG.get("reranker", {})
RERANK_MODEL = RERANK.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOPN = int(RERANK.get("top_n", 30))

AGENT = CFG.get("agent", {})
AUTO_ACTIONS_DEFAULT = bool(AGENT.get("auto_actions", False))

# prefer production pointer if present
ml_model_name = CFG.get("mlflow", {}).get("registered_model", "EKA_RAG_Index")
prod_paths = get_production_index_paths(ml_model_name)
INIT_INDEX_PATH = prod_paths.get("index_path") or CFG_INDEX_PATH
INIT_STORE_PATH = prod_paths.get("store_path") or CFG_STORE_PATH

# models
embedder = Embedder(EMB_MODEL, NORMALIZE)
reranker = Reranker(RERANK_MODEL)

# hot-swappable index with lock
INDEX_LOCK = threading.RLock()
_index = DocIndex(INIT_INDEX_PATH, INIT_STORE_PATH)

def get_index() -> DocIndex:
    with INDEX_LOCK:
        return _index

def swap_index(new_index: DocIndex):
    global _index
    with INDEX_LOCK:
        _index = new_index

# --------- Async index build status ----------
STATUS_LOCK = threading.Lock()
INDEX_LOG: deque = deque(maxlen=200)
INDEX_STATUS: Dict[str, object] = {
    "state": "idle",              # idle|queued|running|done|error|finalizing
    "phase": "",
    "message": "",
    "files_total": 0,
    "files_done": 0,
    "chunks_total": 0,
    "rows_total": 0,
    "rows_done": 0,
    "progress_pct": 0,
    "started_at": None,
    "finished_at": None,
    "index_version": int(time.time()),
    "error": "",
    "last_event_at": None,
}
def _log(ev: str, detail: Dict):
    ts = int(time.time())
    INDEX_LOG.append({"t": ts, "event": ev, **(detail or {})})
    with STATUS_LOCK:
        INDEX_STATUS["last_event_at"] = ts

def _set_status(**kv):
    with STATUS_LOCK:
        INDEX_STATUS.update(kv)

def _compute_pct() -> int:
    st = INDEX_STATUS
    if st["state"] in ("idle", "queued"): return 1 if st["state"] == "queued" else 0
    if st["phase"] == "scanning": return 5
    if st["phase"] == "chunking":
        if st["files_total"] > 0:
            return min(25, max(6, int(25 * st["files_done"]/max(1, st["files_total"]))))
        return 10
    if st["phase"] == "embedding":
        if st["rows_total"] > 0:
            return min(85, max(26, 25 + int(60 * st["rows_done"]/max(1, st["rows_total"]))))
        return 30
    if st["phase"] == "writing": return 90
    if st["phase"] == "finalizing": return 96
    if st["phase"] == "done": return 100
    if st["phase"] == "error": return st.get("progress_pct", 0)
    return st.get("progress_pct", 0)

def _start_rebuild_async():
    with STATUS_LOCK:
        if INDEX_STATUS["state"] in ("queued", "running", "finalizing"):
            return False
        INDEX_STATUS.update({
            "state": "queued", "phase": "scanning", "message": "Queued",
            "files_total": 0, "files_done": 0, "chunks_total": 0,
            "rows_total": 0, "rows_done": 0,
            "progress_pct": 1, "started_at": int(time.time()),
            "finished_at": None, "error": ""
        })
    _log("queued", {})
    t = threading.Thread(target=_run_rebuild, daemon=True)
    t.start()
    return True

def _run_rebuild():
    try:
        _set_status(state="running", phase="scanning", message="Scanning files")
        _set_status(progress_pct=_compute_pct()); _log("phase", {"phase": "scanning"})

        def cb(ev: str, pl: Dict):
            if ev == "scan_total":
                _set_status(files_total=int(pl.get("files", 0)), files_done=0)
                _set_status(message="Files discovered"); _log("scan_total", pl)
            elif ev == "scan_file":
                _log("scan_file", pl)
            elif ev == "chunk_progress":
                _set_status(phase="chunking",
                            files_done=int(pl.get("files_done", 0)),
                            files_total=int(pl.get("files_total", 0)),
                            chunks_total=int(pl.get("chunks", 0)),
                            message=f"Chunked {pl.get('files_done',0)}/{pl.get('files_total',0)} files")
                _log("chunk_progress", pl)
            elif ev == "embed_progress":
                _set_status(phase="embedding",
                            rows_done=int(pl.get("rows_done", 0)),
                            rows_total=int(pl.get("rows_total", 0)),
                            message=f"Embedding {pl.get('rows_done',0)}/{pl.get('rows_total',0)} chunks")
                _log("embed_progress", pl)
            elif ev == "write_index":
                _set_status(phase="writing", message="Writing FAISS index & store"); _log("write_index", {})
            elif ev == "done":
                _set_status(phase="finalizing", message="Finalizing"); _log("done_event", pl)
            _set_status(progress_pct=_compute_pct())

        stats = build_index_fn(progress_cb=cb)

        # Always reload the freshly built index and hot-swap
        fresh_idx = DocIndex(CFG_INDEX_PATH, CFG_STORE_PATH)
        swap_index(fresh_idx)

        _set_status(state="done", phase="done", finished_at=int(time.time()),
                    index_version=int(time.time()),
                    message=f"Indexed {stats.get('num_chunks',0)} chunks from {stats.get('num_files',0)} files (swapped to fresh build)",
                    progress_pct=100)
        _log("complete", {**(stats or {}), "swapped_paths": {"index": CFG_INDEX_PATH, "store": CFG_STORE_PATH}})
    except Exception as e:
        tb = traceback.format_exc()
        _set_status(state="error", phase="error", finished_at=int(time.time()),
                    error=str(e), message="Indexing failed", progress_pct=_compute_pct())
        pathlib.Path("data/eval/last_index_error.log").write_text(tb, encoding="utf-8")
        _log("error", {"error": str(e)})

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Enterprise Knowledge Agent",
    version="0.8.2",
    description="Local agent with hybrid retrieval, reranker, GPT answers, detailed async indexing.",
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Schemas
# ---------------------------
class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    mode: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: Optional[bool] = True

class AnswerRequest(BaseModel):
    question: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    max_chars: Optional[int] = 800
    mode: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: Optional[bool] = True

class UpsertWikiRequest(BaseModel):
    title: str
    content: str

class CreateTicketRequest(BaseModel):
    title: str
    body: str
    priority: str = Field("medium", pattern="^(low|medium|high)$")

class AgentRunRequest(BaseModel):
    message: str
    auto_actions: Optional[bool] = None

# ---------------------------
# Helpers
# ---------------------------
def _smart_preview(text: str, limit: int = 280) -> str:
    t = (text or "").strip()
    if len(t) <= limit: return t
    cut = t[:limit]
    i = max(cut.rfind(" "), cut.rfind("\n"))
    if i > limit // 2:
        cut = cut[:i]
    return cut.rstrip() + "…"

def ensure_ticket_db(db_path: str = "data/tickets.sqlite"):
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
    conn.close()

def _format_citations(results: List[Tuple[Dict, float]]) -> List[dict]:
    cites = []
    for rank, (r, score) in enumerate(results, start=1):
        txt = (r.get("text") or "")
        preview = _smart_preview(txt, 280)
        cites.append({
            "rank": rank,
            "source": r.get("source"),
            "score": float(score),
            "preview": preview,
            "char_count": len(txt),
        })
    return cites

import re as _re
_SENT_SPLIT = _re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')
def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t: return []
    parts = _SENT_SPLIT.split(t)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) < 35 and buf:
            buf = f"{buf} {p}"
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out

def _attribute(answer: str, records: List[Dict]) -> List[dict]:
    sents = _split_sentences(answer)
    if not sents or not records: return []
    sent_vecs = embedder.encode(sents)
    rec_vecs = embedder.encode([(r.get("text") or "") for r in records])

    import numpy as np
    sim = (sent_vecs @ rec_vecs.T).astype(float)
    out = []
    for i, s in enumerate(sents):
        j = int(np.argmax(sim[i]))
        best = records[j]
        out.append({
            "sentence": s,
            "best_source": best.get("source"),
            "best_rank": j + 1,
            "score": float(sim[i, j]),
            "preview": _smart_preview((best.get("text") or ""), 240)
        })
    return out

# ---------- retrieval shaping: wiki boost + guarantee + per-source cap ----------
def _cap_by_source(pairs: List[Tuple[Dict, float]], k: int, per_source: int = 3) -> List[Tuple[Dict, float]]:
    by_src, out = {}, []
    for rec, sc in pairs:
        src = rec.get("source") or ""
        cnt = by_src.get(src, 0)
        if cnt < per_source:
            out.append((rec, sc))
            by_src[src] = cnt + 1
        if len(out) >= k:
            break
    return out

def _apply_wiki_boost(pairs: List[Tuple[Dict, float]], boost: float = 0.35) -> List[Tuple[Dict, float]]:
    boosted = []
    for rec, sc in pairs:
        src = (rec.get("source") or "").replace("\\", "/")
        boosted.append((rec, float(sc) + (boost if "/wiki/" in src else 0.0)))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

def _inject_at_least_one_wiki(pairs: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    has_wiki = any("/wiki/" in (r.get("source","").replace("\\","/")) for r,_ in pairs[:5])
    if has_wiki:
        return pairs
    # pick best wiki in whole pool and insert at position 2 if exists
    best_idx, best_score = -1, float("-inf")
    for i, (rec, sc) in enumerate(pairs):
        if "/wiki/" in (rec.get("source","").replace("\\","/")) and sc > best_score:
            best_idx, best_score = i, sc
    if best_idx >= 0:
        rec = pairs.pop(best_idx)
        pairs.insert(1 if len(pairs) >= 2 else 0, rec)
    return pairs

def _retrieve(question: str, k: int, mode: str, use_rerank: bool):
    t0 = time.time()
    qv = embedder.encode([question])
    idx = get_index()

    # collect a larger candidate pool first
    k_pool = max(k * 6, RERANK_TOPN)

    if mode == "dense":
        raw = idx.query_dense(qv, k=k_pool); meta_mode = "dense"
    elif mode == "sparse":
        raw = idx.query_sparse(question, k=k_pool); meta_mode = "sparse"
    else:
        raw = idx.query_hybrid(qv, question, k=k_pool, alpha=ALPHA); meta_mode = "hybrid"

    candidates = [r for r, _ in raw]
    reranked = raw
    rerank_used = False
    if use_rerank and candidates:
        pairs = reranker.rerank(question, candidates, top_k=k_pool)
        reranked = [(rec, sc) for rec, sc in pairs]
        rerank_used = True

    # boost wiki, ensure one wiki appears, cap per source, trim
    reranked = _apply_wiki_boost(reranked, boost=0.35)
    reranked = _inject_at_least_one_wiki(reranked)
    reranked = _cap_by_source(reranked, k=max(k, 1), per_source=3)
    top_records = [r for r, _ in reranked[:k]]

    meta = {
        "mode": meta_mode,
        "alpha": ALPHA,
        "rerank_used": rerank_used,
        "retrieval_ms": int((time.time() - t0) * 1000),
        "candidates": len(candidates),
        "index_version": INDEX_STATUS.get("index_version"),
    }
    return top_records, reranked, meta

# ---------------------------
# Endpoints
# ---------------------------
class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    mode: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: Optional[bool] = True

class AnswerRequest(BaseModel):
    question: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    max_chars: Optional[int] = 800
    mode: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: Optional[bool] = True

class UpsertWikiRequest(BaseModel):
    title: str
    content: str

class CreateTicketRequest(BaseModel):
    title: str
    body: str
    priority: str = Field("medium", pattern="^(low|medium|high)$")

class AgentRunRequest(BaseModel):
    message: str
    auto_actions: Optional[bool] = None

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    top_records, reranked, meta = _retrieve(req.query, req.k, req.mode, req.rerank)
    return {"query": req.query, "k": req.k, **meta, "results": _format_citations(reranked[:req.k])}

@app.post("/answer_with_citations")
def answer_with_citations(req: AnswerRequest):
    top_records, reranked, meta = _retrieve(req.question, req.k, req.mode, req.rerank)
    answer, llm_meta = generate_answer(top_records, req.question, max_chars=req.max_chars or 800)
    attribution = _attribute(answer, top_records)
    return {"question": req.question,
            "answer": answer,
            "citations": _format_citations(reranked[:req.k]),
            "attribution": attribution,
            "meta": {**meta, **llm_meta}}

@app.post("/agent_run")
def agent_run_post(req: AgentRunRequest):
    return _agent_execute(req.message, req.auto_actions)

@app.get("/agent_run")
def agent_run_get(message: str = Query(...), auto_actions: Optional[bool] = Query(False)):
    return _agent_execute(message, auto_actions)

def _agent_execute(message: str, auto_actions: Optional[bool]):
    auto = AUTO_ACTIONS_DEFAULT if auto_actions is None else bool(auto_actions)
    result = run_agent(message, auto_actions=auto)

    performed = []
    if auto:
        actions = []
        for step in result.get("trace", []):
            if step.get("action") == "critic" and step.get("actions"):
                actions.extend(step["actions"])
        for act in actions:
            try:
                if act.get("type") == "upsert_wiki_draft":
                    title = act.get("title") or "agent-draft"
                    content = act.get("content") or "# Draft\n\n(Empty content)"
                    resp = upsert_wiki_page(UpsertWikiRequest(title=title, content=content))
                    performed.append({
                        "type": "upsert_wiki_draft",
                        "title": title,
                        "slug": resp.get("slug"),
                        "path": resp.get("path"),
                        "reindex_queued": True
                    })
            except Exception as e:
                performed.append({"type": act.get("type"), "error": str(e)})

    return {"status": "ok", "auto_actions": auto, **result, "performed_actions": performed}

@app.post("/upsert_wiki_page")
def upsert_wiki_page(req: UpsertWikiRequest):
    wiki_dir = pathlib.Path("data/processed/wiki"); wiki_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(req.title) or "page"
    fp = wiki_dir / f"{slug}.md"
    fp.write_text(req.content, encoding="utf-8")
    _start_rebuild_async()
    return {"status": "ok", "slug": slug, "path": str(fp)}

@app.post("/create_ticket")
def create_ticket(req: CreateTicketRequest):
    ensure_ticket_db()
    conn = sqlite3.connect("data/tickets.sqlite")
    with conn:
        cur = conn.execute("INSERT INTO tickets(title, body, priority) VALUES (?,?,?)",
                           (req.title, req.body, req.priority))
        tid = cur.lastrowid
        row = conn.execute("SELECT id,title,priority,status,created_at FROM tickets WHERE id=?",(tid,)).fetchone()
    conn.close()
    return {"status":"ok","ticket":{"id":row[0],"title":row[1],"priority":row[2],"status":row[3],"created_at":row[4]}}

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    content = await file.read()
    name = title or file.filename or "uploaded"
    ext = pathlib.Path(file.filename or "").suffix.lower()

    if ext in [".md", ".txt"]:
        text = content.decode("utf-8", errors="ignore")
    else:
        bin_dir = pathlib.Path("data/raw/uploads"); bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / (file.filename or "upload.bin"); bin_path.write_bytes(content)
        text = f"# Attachment: {file.filename}\n\nStored at: {bin_path}\n\n(Consider adding a PDF→text converter.)"

    wiki_dir = pathlib.Path("data/processed/wiki"); wiki_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(name) or "page"
    md_path = wiki_dir / f"{slug}.md"; md_path.write_text(text, encoding="utf-8")

    queued = _start_rebuild_async()
    return {"status":"ok","stored_as":str(md_path),"slug":slug,"reindex_queued":queued}

@app.get("/index_status")
def index_status():
    with STATUS_LOCK:
        st = dict(INDEX_STATUS)
        st["progress_pct"] = _compute_pct()
        st["log_tail"] = list(INDEX_LOG)[-40:]
        return st

@app.post("/rebuild_index")
def rebuild_index():
    queued = _start_rebuild_async()
    return {"queued": queued}

@app.get("/healthz")
def healthz():
    with STATUS_LOCK:
        return {
            "status": "ok",
            "ready": True,
            "embedder_model": EMB_MODEL,
            "index_path": str(CFG_INDEX_PATH),
            "store_path": str(CFG_STORE_PATH),
            "index_version": INDEX_STATUS.get("index_version"),
            "index_state": INDEX_STATUS.get("state"),
        }
