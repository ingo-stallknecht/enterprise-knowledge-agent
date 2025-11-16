# app/llm/answerer.py

from typing import List, Dict, Tuple
import os
import textwrap

# Optional: we try to access Streamlit secrets if available (for Streamlit Cloud)
try:
    import streamlit as st  # type: ignore
except Exception:  # running in tests / non-Streamlit context
    st = None

try:
    # New OpenAI client (>=1.0)
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None


# -------------------------------------------------------------------
# Helpers to hydrate env from Streamlit secrets if available
# -------------------------------------------------------------------

def _hydrate_openai_env_from_secrets() -> None:
    """
    Ensure OPENAI_API_KEY and USE_OPENAI are set in os.environ if possible.

    This mirrors the logic you used in streamlit_app.py but makes the
    answerer self-contained, so it also works if imports / init order
    are different on Streamlit Cloud.
    """
    # 1) USE_OPENAI: default to "true" unless explicitly disabled
    if os.environ.get("USE_OPENAI") is None:
        val = "true"
        if st is not None:
            try:
                raw = st.secrets.get("USE_OPENAI", None)
                if raw is not None:
                    val = str(raw).lower()
            except Exception:
                pass
        os.environ["USE_OPENAI"] = "true" if val == "true" else "false"

    # 2) OPENAI_API_KEY: if missing in env, try to pull from secrets
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return

    if st is None:
        return

    key = ""
    try:
        # Direct keys at root
        for k in ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_APIKEY", "openai_api_key", "openai_key"]:
            if k in st.secrets:
                v = str(st.secrets[k]).strip()
                if v:
                    key = v
                    break

        # Nested sections
        if not key:
            for section in ("openai", "OPENAI", "llm"):
                if section in st.secrets and isinstance(st.secrets[section], dict):
                    sec = st.secrets[section]
                    for k in ("api_key", "API_KEY", "key"):
                        v = str(sec.get(k, "")).strip()
                        if v:
                            key = v
                            break
                    if key:
                        break
    except Exception:
        key = ""

    if key:
        os.environ["OPENAI_API_KEY"] = key


def _openai_enabled() -> bool:
    """
    Check if we should attempt to use OpenAI.

    Conditions:
    - USE_OPENAI=true (env, with default handled by _hydrate_openai_env_from_secrets)
    - OPENAI_API_KEY present and non-empty
    - openai client import succeeded
    """
    if _OpenAI is None:
        return False

    # Make sure env is hydrated from secrets if available
    _hydrate_openai_env_from_secrets()

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
            client = _OpenAI()  # key read from env (OPENAI_API_KEY)

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
