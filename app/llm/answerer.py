# app/llm/answerer.py
import os
from typing import List, Dict
from app.llm.gpt_client import GPTClient

def simple_extractive_answer(chunks: List[Dict], max_chars: int = 800) -> str:
    acc, total = [], 0
    for rec in chunks:
        t = (rec.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) + 1 > max_chars:
            if max_chars - total > 0:
                acc.append(t[: max_chars - total])
            break
        acc.append(t)
        total += len(t) + 1
        if total >= max_chars:
            break
    return "\n\n".join(acc).strip()

_gpt = None
def _get_gpt():
    global _gpt
    if _gpt is None:
        _gpt = GPTClient()
    return _gpt

def generate_answer(chunks: List[Dict], question: str, max_chars: int = 800) -> str:
    """If USE_OPENAI=true and under budget → GPT summary; else fallback extractive."""
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
    if not use_openai:
        return simple_extractive_answer(chunks, max_chars=max_chars)

    try:
        ctx = "\n\n".join((c.get("text") or "") for c in chunks)  # server trims further
        gpt = _get_gpt()
        ans, _meta = gpt.answer(ctx, question)
        return ans
    except Exception:
        # Any failure (no key, budget exceeded, network) → graceful fallback
        return simple_extractive_answer(chunks, max_chars=max_chars)
