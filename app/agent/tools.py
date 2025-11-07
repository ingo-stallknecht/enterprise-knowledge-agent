# app/agent/tools.py
import os
import json
import sqlite3
import pathlib
from typing import Dict, Any, List
from slugify import slugify

DATA = pathlib.Path("data")
WIKI_DIR = DATA / "processed" / "wiki"
LOG_DIR = DATA / "agent"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR.mkdir(parents=True, exist_ok=True)

def create_wiki(title: str, content: str) -> Dict[str, Any]:
    slug = slugify(title) or "page"
    fp = WIKI_DIR / f"{slug}.md"
    fp.write_text(content, encoding="utf-8")
    _log_tool("create_wiki", {"title": title, "path": str(fp)})
    return {"type": "wiki", "title": title, "path": str(fp), "slug": slug}

def open_ticket(title: str, body: str, priority: str = "medium") -> Dict[str, Any]:
    db = DATA / "tickets.sqlite"
    conn = sqlite3.connect(str(db))
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
        cur = conn.execute("INSERT INTO tickets(title, body, priority) VALUES (?,?,?)", (title, body, priority))
        ticket_id = cur.lastrowid
        row = conn.execute("SELECT id, title, priority, status, created_at FROM tickets WHERE id=?", (ticket_id,)).fetchone()
    conn.close()
    out = {"type": "ticket", "id": row[0], "title": row[1], "priority": row[2], "status": row[3], "created_at": row[4]}
    _log_tool("open_ticket", out)
    return out

def summarize_uploads() -> Dict[str, Any]:
    # very simple: list newest wiki files with first 200 chars
    items: List[Dict[str, str]] = []
    for fp in sorted(WIKI_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        txt = fp.read_text(encoding="utf-8")
        items.append({"file": fp.name, "preview": (txt[:200] + ("…" if len(txt) > 200 else ""))})
    out = {"type": "uploads_summary", "items": items}
    _log_tool("summarize_uploads", out)
    return out

def _log_tool(name: str, payload: Dict[str, Any]) -> None:
    rec = {"tool": name, "payload": payload}
    with (LOG_DIR / "tool_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
