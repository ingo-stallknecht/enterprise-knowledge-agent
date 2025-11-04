# app/rag/evaluator.py
from typing import Dict, List
import json, re


class SimpleEvaluator:
def __init__(self, retriever, k: int):
self.retriever = retriever
self.k = k


def eval_questions(self, questions: List[Dict]):
hits, total = 0, 0
contains, total_q = 0, 0
for q in questions:
res = self.retriever.retrieve(q["question"], k=self.k)
total += 1
# recall@k if any gold token appears in any chunk (proxy)
gold = [g.lower() for g in q.get("gold", [])]
hay = "\n".join([r["text"].lower() for r in res])
if any(g in hay for g in gold):
hits += 1
# answer_contains_gold_frac: same measure
if gold:
contains += 1 if any(g in hay for g in gold) else 0
total_q += 1
recall_at_k = hits / max(1, total)
answer_contains_gold_frac = contains / max(1, total_q)
return {"recall_at_k": recall_at_k, "answer_contains_gold_frac": answer_contains_gold_frac}