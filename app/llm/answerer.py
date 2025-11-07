# app/llm/answerer.py
import os
from typing import List, Dict, Tuple
from app.llm.gpt_client import GPTClient

def simple_extractive_answer(chunks: List[Dict], max_chars: int = 800) -> str:
    acc, total = [], 0
    for rec in chunks:
        t = (rec.get("text") or "").strip()
        if not t: continue
        if total + len(t) + 1 > max_chars:
            acc.append(t[: max_chars - total]); break
        acc.append(t); total += len(t) + 1
        if total >= max_chars: break
    return "\n\n".join(acc).strip()

_gpt = None
def _get_gpt():
    global _gpt
    if _gpt is None: _gpt = GPTClient()
    return _gpt

def generate_answer(chunks: List[Dict], question: str, max_chars: int = 800) -> Tuple[str, dict]:
    """GPT if enabled; otherwise fallback extractive. Returns (answer, meta)."""
    if os.getenv("USE_OPENAI", "false").lower() != "true":
        return simple_extractive_answer(chunks, max_chars), {"llm": "disabled"}
    try:
        ctx = "\n\n".join((c.get("text") or "") for c in chunks)
        gpt = _get_gpt()
        ans, meta = gpt.answer(ctx, question)
        return ans, {"llm": gpt.model, **meta}
    except Exception as e:
        return simple_extractive_answer(chunks, max_chars), {"llm": "fallback", "error": str(e)}
