# app/llm/answerer.py
import os
from typing import List, Dict, Tuple
from app.llm.gpt_client import GPTClient

FALLBACK_MSG = "No relevant context found in the current corpus."

def simple_extractive_answer(chunks: List[Dict], max_chars: int = 900) -> Tuple[str, dict]:
    if not chunks:
        return FALLBACK_MSG, {"llm": "extractive"}
    acc, total = [], 0
    for rec in chunks:
        t = (rec.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) + 1 > max_chars:
            # add trimmed remainder and stop
            remaining = max_chars - total
            if remaining > 0:
                acc.append(t[:remaining])
            break
        acc.append(t)
        total += len(t) + 1
        if total >= max_chars:
            break
    text = "\n\n".join(acc).strip()
    if not text:
        text = FALLBACK_MSG
    return text, {"llm": "extractive"}

_gpt = None
def _get_gpt():
    global _gpt
    if _gpt is None:
        _gpt = GPTClient()
    return _gpt

def generate_answer(chunks: List[Dict], question: str, max_chars: int = 900) -> Tuple[str, dict]:
    use_gpt = os.getenv("USE_OPENAI", "false").lower() == "true"
    # If no chunks, prefer an explicit message rather than a hallucinated GPT answer
    if not chunks:
        if use_gpt:
            # still ask GPT to state "Not in context."
            try:
                gpt = _get_gpt()
                ans, meta = gpt.answer("", question)
                meta["llm"] = meta.get("llm", "gpt")
                return ans or "Not in context.", meta
            except Exception:
                return FALLBACK_MSG, {"llm": "extractive"}
        return FALLBACK_MSG, {"llm": "extractive"}

    if not use_gpt:
        return simple_extractive_answer(chunks, max_chars)

    # Try GPT; if it fails or returns empty, fall back to extractive
    try:
        ctx = "\n\n".join((c.get("text") or "") for c in chunks)
        gpt = _get_gpt()
        ans, meta = gpt.answer(ctx, question)
        ans = (ans or "").strip()
        if not ans:
            return simple_extractive_answer(chunks, max_chars)
        meta["llm"] = meta.get("llm", "gpt")
        # hard cap output length without cutting words mid-sentence too brutally
        if len(ans) > max_chars:
            ans = ans[:max_chars].rstrip()
        return ans, meta
    except Exception:
        return simple_extractive_answer(chunks, max_chars)
