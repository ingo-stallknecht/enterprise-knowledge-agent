# app/server.py
import os
import sys
import pathlib
import sqlite3
import subprocess
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
import requests

# Make local packages importable when running `uvicorn app.server:app`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slugify import slugify
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.utils import ensure_dirs, load_cfg, get_production_index_paths
from app.llm.answerer import generate_answer  # GPT-backed (optional) with safe fallback


# ---------------------------
# Setup & config
# ---------------------------
ensure_dirs()
CFG_PATH = "configs/settings.yaml"
cfg = load_cfg(CFG_PATH)

retrieval_cfg = cfg.get("retrieval", {})
embedder_model = retrieval_cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
normalize = bool(retrieval_cfg.get("normalize", True))
TOP_K_DEFAULT = int(retrieval_cfg.get("top_k", 6))

# Prefer Production index from MLflow if available
ml_model_name = cfg.get("mlflow", {}).get("registered_model", "EKA_RAG_Index")
prod_paths = get_production_index_paths(ml_model_name)
if prod_paths and prod_paths.get("index_path") and prod_paths.get("store_path"):
    index_path = prod_paths["index_path"]
    store_path = prod_paths["store_path"]
else:
    index_path = retrieval_cfg.get("faiss_index", "data/index/handbook.index")
    store_path = retrieval_cfg.get("store_json", "data/index/docstore.json")

embedder = Embedder(embedder_model, normalize)
index = DocIndex(index_path, store_path)


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Enterprise Knowledge Agent (RAG + Actions)",
    version="0.1.0",
    description="Local, no-cloud portfolio variant. RAG over GitLab Handbook with action endpoints.",
)


# ---------------------------
# Schemas
# ---------------------------
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=20)

class AnswerRequest(BaseModel):
    question: str = Field(..., description="Question to answer using RAG")
    k: Optional[int] = Field(TOP_K_DEFAULT, ge=1, le=20)
    max_chars: Optional[int] = 800

class UpsertWikiRequest(BaseModel):
    title: str
    content: str

class CreateTicketRequest(BaseModel):
    title: str
    body: str
    priority: str = Field("medium", pattern="^(low|medium|high)$")

class AgentRequest(BaseModel):
    message: str = Field(..., description="Natural-language instruction")
    k: Optional[int] = TOP_K_DEFAULT
    max_chars: Optional[int] = 800


# ---------------------------
# Helpers
# ---------------------------
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

def format_citations(results: List[tuple]) -> List[dict]:
    cites = []
    for r, score in results:
        cites.append({
            "source": r.get("source"),
            "score": float(score),
            "preview": (r.get("text", "")[:180] + "…") if r.get("text") else ""
        })
    return cites

def rebuild_index_sync():
    """Rebuild index by invoking the existing script synchronously."""
    subprocess.check_call([sys.executable, "-m", "scripts.build_index"])

def save_markdown_doc(title: str, content: str) -> str:
    wiki_dir = pathlib.Path("data/processed/wiki")
    wiki_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title) or "page"
    fp = wiki_dir / f"{slug}.md"
    fp.write_text(content, encoding="utf-8")
    return str(fp)

def fetch_url_to_markdown(url: str) -> str:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    try:
        from markdownify import markdownify as md
        text = md(r.text)
    except Exception:
        text = r.text
    title = url.split("://", 1)[-1][:80]
    return save_markdown_doc(title, text)

def detect_intent(message: str) -> str:
    m = message.lower()
    if any(x in m for x in ["upload", "attach", "add file", "ingest file"]):
        return "ingest_file"
    if any(x in m for x in ["ingest url", "add url", "fetch url", "crawl "]):
        return "ingest_url"
    if any(x in m for x in ["create ticket", "open ticket", "new ticket", "bug", "issue"]):
        return "create_ticket"
    if any(x in m for x in ["add wiki", "write wiki", "create wiki", "upsert wiki", "document this"]):
        return "upsert_wiki_page"
    if any(x in m for x in ["search", "find", "retrieve", "look up"]):
        return "retrieve"
    if any(x in m for x in ["answer", "explain", "what is", "how do", "how to", "summarize"]):
        return "answer"
    return "answer"


# ---------------------------
# Endpoints (RAG core)
# ---------------------------
@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    qv = embedder.encode([req.query])
    results = index.query(qv, k=req.k)
    payload = [{
        "text": rec.get("text"),
        "source": rec.get("source"),
        "score": float(score),
    } for rec, score in results]
    return {"query": req.query, "k": req.k, "results": payload}

@app.post("/answer_with_citations")
def answer_with_citations(req: AnswerRequest):
    qv = embedder.encode([req.question])
    results = index.query(qv, k=req.k)
    top_records = [r for r, _ in results]
    # GPT (if enabled) for concise answer; otherwise fallback to local extractive
    answer = generate_answer(top_records, req.question, max_chars=req.max_chars or 800)
    return {
        "question": req.question,
        "answer": answer,
        "citations": format_citations(results)
    }

@app.post("/upsert_wiki_page")
def upsert_wiki_page(req: UpsertWikiRequest):
    path = save_markdown_doc(req.title, req.content)
    return {"status": "ok", "slug": slugify(req.title), "path": path}

@app.post("/create_ticket")
def create_ticket(req: CreateTicketRequest):
    ensure_ticket_db()
    conn = sqlite3.connect("data/tickets.sqlite")
    with conn:
        cur = conn.execute(
            "INSERT INTO tickets(title, body, priority) VALUES (?, ?, ?)",
            (req.title, req.body, req.priority),
        )
        ticket_id = cur.lastrowid
        row = conn.execute(
            "SELECT id, title, priority, status, created_at FROM tickets WHERE id=?",
            (ticket_id,)
        ).fetchone()
    conn.close()
    return {
        "status": "ok",
        "ticket": {"id": row[0], "title": row[1], "priority": row[2], "status": row[3], "created_at": row[4]}
    }


# ---------------------------
# Endpoints (Agent & Ingestion)
# ---------------------------
@app.post("/agent")
def agent(req: AgentRequest):
    """
    Natural-language entry point:
    - "Add this URL: https://example.com/..."
    - "Create a ticket: onboarding flow broken, high priority"
    - "Add a wiki 'Onboarding' with bullets: ..."
    - "How do I propose a change to the handbook?"
    """
    intent = detect_intent(req.message)
    if intent == "retrieve":
        qv = embedder.encode([req.message])
        results = index.query(qv, k=req.k)
        return {"intent": intent, "results": format_citations(results)}
    elif intent == "answer":
        qv = embedder.encode([req.message])
        results = index.query(qv, k=req.k)
        ans = generate_answer([r for r, _ in results], req.message, max_chars=req.max_chars or 800)
        return {"intent": intent, "answer": ans, "citations": format_citations(results)}
    elif intent == "create_ticket":
        title = (req.message.split(".")[0] or "New ticket")[:120]
        body = req.message
        return create_ticket(CreateTicketRequest(title=title, body=body, priority="medium"))
    elif intent == "upsert_wiki_page":
        title = "Note from agent"
        if "'" in req.message:
            try:
                title = req.message.split("'", 1)[1].split("'", 1)[0]
            except Exception:
                pass
        path = save_markdown_doc(title, req.message)
        rebuild_index_sync()
        return {"intent": intent, "status": "ok", "path": path}
    elif intent == "ingest_url":
        url = None
        for tok in req.message.split():
            if tok.startswith("http://") or tok.startswith("https://"):
                url = tok.strip()
                break
        if not url:
            return {"intent": intent, "error": "No URL found in message."}
        path = fetch_url_to_markdown(url)
        rebuild_index_sync()
        return {"intent": intent, "status": "ok", "url": url, "path": path}
    else:
        return {"intent": intent, "message": "Intent not supported in this demo."}

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    """
    Upload a TXT/MD/PDF, store it as a wiki page, then rebuild the index.
    """
    content = await file.read()
    name = title or file.filename
    ext = pathlib.Path(file.filename).suffix.lower()
    if ext in [".md", ".txt"]:
        text = content.decode("utf-8", errors="ignore")
    else:
        bin_dir = pathlib.Path("data/raw/uploads"); bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / file.filename
        bin_path.write_bytes(content)
        text = f"# Attachment: {file.filename}\n\nStored at: {bin_path}\n\n(Consider adding PDF→text converter.)"
    md_path = save_markdown_doc(name, text)
    rebuild_index_sync()
    return {"status": "ok", "stored_as": md_path}


# ---------------------------
# Health
# ---------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "index_path": str(index_path), "store_path": str(store_path)}

