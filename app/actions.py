# app/actions.py
import os, pathlib, sqlite3, datetime
from slugify import slugify as _slugify


def slugify(title: str):
return _slugify(title, lowercase=True, separator="-")


WIKI_DIR = pathlib.Path("data/processed/wiki"); WIKI_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = pathlib.Path("data/tickets.db"); DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ensure sqlite table
conn = sqlite3.connect(DB_PATH)
with conn:
conn.execute("CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, body TEXT, priority TEXT, created_at TEXT)")
conn.close()


def upsert_wiki_page(title: str, content: str):
fp = WIKI_DIR / f"{slugify(title)}.md"
fp.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
return {"status": "ok", "path": str(fp)}


def create_ticket(title: str, body: str, priority: str = "medium"):
conn = sqlite3.connect(DB_PATH)
with conn:
conn.execute("INSERT INTO tickets (title, body, priority, created_at) VALUES (?,?,?,?)",
(title, body, priority, datetime.datetime.utcnow().isoformat()+"Z"))
return {"status": "ok", "id": conn.execute("select last_insert_rowid()").fetchone()[0]}