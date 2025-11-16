# app/llm/answerer.py

from typing import List, Dict, Tuple
import os
import textwrap

try:
    # New OpenAI client (>=1.0)
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None


def _openai_enabled() -> bool:
    """
    Check if we should attempt to use OpenAI.

    Conditions:
    - USE_OPENAI=true (env)
    - OPENAI_API_KEY present and non-empty
    - openai client import succeeded
    """
    if _OpenAI is None:
        return False

    if os.environ.get("USE_OPENAI", "false").lower() != "true":
        return False

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return False

    return True


def _build_context(recs: List[Dict], max_chars: int = 6000) -> str:
    """
    Join retrieved chunks into one context string, truncated by characters.
    """
    parts: List[str] = []
    total = 0
    for r in recs:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        if total + len(txt) + 2 > max_chars:
            # Add as much as we can without exploding
            remaining = max_chars - total
            if remaining > 0:
                parts.append(txt[:remaining])
            break
        parts.append(txt)
        total += len(txt) + 2
    return "\n\n".join(parts).strip()


def _extractive_fallback(recs: List[Dict], question: str, max_chars: int) -> Tuple[str, Dict]:
    """
    Simple extractive answer: just stitch together the most relevant chunks.
    """
    joined = "\n\n".join((r.get("text") or "") for r in recs).strip()
    if not joined:
        ans = "No relevant context found in the current corpus."
    else:
        ans = joined[:max_chars]
    meta = {
        "llm": "extractive",
        "used_openai": False,
        "reason": "fallback_or_disabled",
    }
    return ans, meta


def generate_answer(
    recs: List[Dict],
    question: str,
    max_chars: int = 900,
) -> Tuple[str, Dict]:
    """
    Main entry point used by the Streamlit app.

    Args:
        recs: list of document chunks (each with at least 'text')
        question: user question
        max_chars: soft character budget for the final answer

    Returns:
        (answer_text, meta_dict)
    """
    # If no context, short early message
    if not recs:
        return (
            "No relevant context found in the current corpus.",
            {"llm": "extractive", "used_openai": False, "reason": "no_context"},
        )

    # Try OpenAI first if enabled
    if _openai_enabled():
        try:
            client = _OpenAI()  # key read from env

            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
            # Rough heuristic: characters ~= 4 tokens, keep some buffer
            max_tokens = min(900, max(200, max_chars // 3))

            context = _build_context(recs, max_chars=6000)

            system_msg = (
                "You are a helpful assistant answering questions strictly from the provided context. "
                "If the context is insufficient, say so clearly. "
                "Do not hallucinate facts that are not in the context."
            )
            user_msg = textwrap.dedent(
                f"""
                Question:
                {question}

                Context (from internal documentation and wiki pages):
                {context}

                Instructions:
                - Base your answer only on the context above.
                - Cite key concepts in a concise, well-structured way.
                - If something is not covered, explicitly say that the current corpus does not contain it.
                - Keep the answer under approximately {max_chars} characters.
                """
            ).strip()

            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                # If somehow empty, fallback
                return _extractive_fallback(recs, question, max_chars)

            # Hard trim by characters just to be safe
            if len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."

            meta = {
                "llm": model,
                "used_openai": True,
                "reason": "openai_success",
            }
            return text, meta

        except Exception:
            # Any failure: fallback to extractive
            return _extractive_fallback(recs, question, max_chars)

    # If OpenAI disabled/unavailable, always use extractive
    return _extractive_fallback(recs, question, max_chars)
