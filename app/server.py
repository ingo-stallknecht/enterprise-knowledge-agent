from fastapi import FastAPI
from pydantic import BaseModel
from .agent import KnowledgeAgent
from .actions import upsert_wiki_page, create_ticket


app = FastAPI(title="Enterprise Knowledge Agent (Local)")
agent = KnowledgeAgent()


class Query(BaseModel):
query: str
k: int | None = None


class Question(BaseModel):
question: str


class Wiki(BaseModel):
title: str
content: str


class Ticket(BaseModel):
title: str
body: str
priority: str = "medium"


@app.post("/retrieve")
def api_retrieve(q: Query):
return {"results": agent.retrieve(q.query, q.k or None)}


@app.post("/answer_with_citations")
def api_answer(q: Question):
return agent.answer_with_citations(q.question)


@app.post("/upsert_wiki_page")
def api_upsert(w: Wiki):
return upsert_wiki_page(w.title, w.content)


@app.post("/create_ticket")
def api_ticket(t: Ticket):
return create_ticket(t.title, t.body, t.priority)