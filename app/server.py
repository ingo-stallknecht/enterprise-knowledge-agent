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

import sys, pathlib, sqlite3, time, re, threading, traceback
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slugify import slugify
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import ensure_dirs, load_cfg, get_production_index_paths
from app.llm.answerer import generate_answer
from app.rag.reranker import Reranker
from app.agent.planner import run_agent
from scripts.build_index import build_index as build_index_fn

# ---------- Init / Config ----------
ensure_dirs()
CFG = load_cfg("configs/settings.yaml")

RET = CFG.get("retrieval", {})
EMB_MODEL = RET.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
NORMALIZE = bool(RET.get("normalize", True))
TOP_K_DEFAULT = int(RET.get("top_k", 12))
ALPHA = float(RET.get("hybrid_alpha", 0.6))

RERANK = CFG.get("reranker", {})
RERANK_MODEL = RERANK.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOPN = int(RERANK.get("top_n", 30))

AGENT = CFG.get("agent", {})
AUTO_ACTIONS_DEFAULT = bool(AGENT.get("auto_actions", False))

ml_model_name = CFG.get("mlflow", {}).get("registered_model", "EKA_RAG_Index")
prod_paths = get_production_index_paths(ml_model_name)
if prod_paths and prod_paths.get("index_path") and prod_paths.get("store_path"):
    INDEX_PATH = prod_paths["index_path"]
    STORE_PATH = prod_paths["store_path"]
else:
    INDEX_PATH = RET.get("faiss_index", "data/index/handbook.index")
    STORE_PATH = RET.get("store_json", "data/index/docstore.json")

embedder = Embedder(EMB_MODEL, NORMALIZE)
reranker = Reranker(RERANK_MODEL)

INDEX_LOCK = threading.RLock()
_index = DocIndex(INDEX_PATH, STORE_PATH)

def get_index() -> DocIndex:
    with INDEX_LOCK:
        return _index

def swap_index(new_index: DocIndex):
    global _index
    with INDEX_LOCK:
        _index = new_index

# --------- Async index build status ----------
STATUS_LOCK = threading.Lock()
INDEX_STATUS: Dict[str, object] = {
    "state": "idle",              # idle|queued|running|done|error
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
}
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
        if INDEX_STATUS["state"] in ("queued", "running"):
            return False
        INDEX_STATUS.update({
            "state": "queued", "phase": "scanning", "message": "Queued",
            "files_total": 0, "files_done": 0, "chunks_total": 0,
            "rows_total": 0, "rows_done": 0,
            "progress_pct": 1, "started_at": int(time.time()),
            "finished_at": None, "error": ""
        })
    t = threading.Thread(target=_run_rebuild, daemon=True)
    t.start()
    return True

def _run_rebuild():
    try:
        _set_status(state="running", phase="scanning", message="Scanning files"); _set_status(progress_pct=_compute_pct())

        def cb(ev: str, pl: Dict):
            if ev == "scan_total":
                _set_status(files_total=int(pl.get("files", 0)), files_done=0, message="Files discovered")
            elif ev == "chunk_progress":
                _set_status(phase="chunking",
                            files_done=int(pl.get("files_done", 0)),
                            files_total=int(pl.get("files_total", 0)),
                            chunks_total=int(pl.get("chunks", 0)),
                            message=f"Chunked {pl.get('files_done',0)}/{pl.get('files_total',0)} files")
            elif ev == "embed_progress":
                _set_status(phase="embedding",
                            rows_done=int(pl.get("rows_done", 0)),
                            rows_total=int(pl.get("rows_total", 0)),
                            message=f"Embedding {pl.get('rows_done',0)}/{pl.get('rows_total',0)} chunks")
            elif ev == "write_index":
                _set_status(phase="writing", message="Writing FAISS index & store")
            elif ev == "done":
                _set_status(phase="finalizing", message="Finalizing")
            _set_status(progress_pct=_compute_pct())

        stats = build_index_fn(progress_cb=cb)

        new_idx = DocIndex(INDEX_PATH, STORE_PATH)
        swap_index(new_idx)

        _set_status(state="done",
                    phase="done",
                    finished_at=int(time.time()),
                    index_version=int(time.time()),
                    message=f"Indexed {stats.get('num_chunks',0)} chunks from {stats.get('num_files',0)} files",
                    progress_pct=100)
    except Exception as e:
        tb = traceback.format_exc()
        _set_status(state="error",
                    phase="error",
                    finished_at=int(time.time()),
                    error=str(e),
                    message="Indexing failed",
                    progress_pct=_compute_pct())
        pathlib.Path("data/eval/last_index_error.log").write_text(tb, encoding="utf-8")

# ---------- FastAPI app & CORS ----------
app = FastAPI(
    title="Enterprise Knowledge Agent",
    version="0.8.0",
    description="Hybrid RAG + reranker + GPT answers + async indexing + promotion/rollback.",
)

# CORS for Streamlit Cloud and local dev
ALLOWED_ORIGINS = [
    "https://*.streamlit.app",
    "https://*.streamlit.io",
    "http://localhost",
    "http://127.0.0.1",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # for quick testing you can use ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    mode: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: Optional[bool] = True

class AnswerRequest(BaseModel):
    question: str
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=50)
    max_chars: Optional[int] = 900
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

# ---------- Helpers ----------
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
        )""")
    conn.close()

def _format_citations(results: List[Tuple[Dict, float]]) -> List[dict]:
    cites = []
    for rank, (r, score) in enumerate(results, start=1):
        txt = (r.get("text") or "").strip()
        preview = (txt[:280] + "…") if len(txt) > 280 else txt
        cites.append({"rank": rank, "source": r.get("source"), "score": float(score),
                      "preview": preview, "char_count": len(txt)})
    return cites

_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')
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
        out.append({"sentence": s, "best_source": best.get("source"),
                    "best_rank": j + 1, "score": float(sim[i, j]),
                    "preview": (best.get("text") or "")[:240] + ("…" if len((best.get("text") or "")) > 240 else "")})
    return out

def _retrieve(question: str, k: int, mode: str, use_rerank: bool):
    t0 = time.time()
    qv = embedder.encode([question])
    idx = get_index()
    if mode == "dense":
        raw = idx.query_dense(qv, k=max(k, RERANK_TOPN)); meta_mode = "dense"
    elif mode == "sparse":
        raw = idx.query_sparse(question, k=max(k, RERANK_TOPN)); meta_mode = "sparse"
    else:
        raw = idx.query_hybrid(qv, question, k=max(k, RERANK_TOPN), alpha=ALPHA); meta_mode = "hybrid"

    candidates = [r for r, _ in raw]
    reranked = raw
    rerank_used = False
    if use_rerank and candidates:
        pairs = reranker.rerank(question, candidates, top_k=max(k, 1))
        reranked = [(rec, sc) for rec, sc in pairs]
        rerank_used = True

    top_records = [r for r, _ in reranked[:k]]
    meta = {"mode": meta_mode, "alpha": ALPHA, "rerank_used": rerank_used,
            "retrieval_ms": int((time.time() - t0) * 1000),
            "candidates": len(candidates), "index_version": INDEX_STATUS.get("index_version")}
    return top_records, reranked, meta

# ---------- Endpoints ----------
@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    top_records, reranked, meta = _retrieve(req.query, req.k, req.mode, req.rerank)
    return {"query": req.query, "k": req.k, **meta, "results": _format_citations(reranked[:req.k])}

@app.post("/answer_with_citations")
def answer_with_citations(req: AnswerRequest):
    top_records, reranked, meta = _retrieve(req.question, req.k, req.mode, req.rerank)
    answer, llm_meta = generate_answer(top_records, req.question, max_chars=req.max_chars or 900)
    attribution = _attribute(answer, top_records)
    return {"question": req.question, "answer": answer,
            "citations": _format_citations(reranked[:req.k]),
            "attribution": attribution, "meta": {**meta, **llm_meta}}

@app.post("/agent_run")
def agent_run(req: AgentRunRequest):
    auto = AUTO_ACTIONS_DEFAULT if req.auto_actions is None else bool(req.auto_actions)
    result = run_agent(req.message, auto_actions=auto)
    return {"status": "ok", "auto_actions": auto, **result}

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
            "index_path": str(INDEX_PATH),
            "store_path": str(STORE_PATH),
            "index_version": INDEX_STATUS.get("index_version"),
            "index_state": INDEX_STATUS.get("state"),
        }
