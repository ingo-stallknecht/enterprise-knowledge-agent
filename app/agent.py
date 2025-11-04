# app/agent.py
from typing import List, Dict
from transformers import pipeline
from .rag.retriever import Retriever
import yaml


cfg = yaml.safe_load(open("configs/settings.yaml"))


class KnowledgeAgent:
def __init__(self):
self.retriever = Retriever(
cfg["retrieval"]["embedder_model"],
cfg["retrieval"]["normalize"],
cfg["retrieval"]["faiss_index"],
cfg["retrieval"]["store_json"],
)
strat = cfg["answer"]["strategy"]
if strat == "summarization":
self.summarizer = pipeline("summarization", model=cfg["answer"]["model_name"]) # downloads once
else:
self.summarizer = None


def retrieve(self, query: str, k: int = None):
return self.retriever.retrieve(query, k or cfg["retrieval"]["top_k"])


def answer_with_citations(self, question: str) -> Dict:
docs = self.retrieve(question)
context = "\n\n".join(d["text"] for d in docs)
if self.summarizer:
summary = self.summarizer(context, max_length=cfg["answer"]["max_tokens"], min_length=60, do_sample=False)[0]["summary_text"]
else:
# naive extractive fallback: first N sentences
summary = "\n".join([d["text"][:300] for d in docs])
citations = [{"source": d["source"], "score": d["score"]} for d in docs]
return {"answer": summary, "citations": citations}