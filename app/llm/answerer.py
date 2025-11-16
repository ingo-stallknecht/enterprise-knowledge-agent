# app/llm/answerer.py

from typing import List, Dict, Tuple
import os
import textwrap

# Optional: Streamlit secrets (for Streamlit Cloud)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

try:
    # New OpenAI client (>=1.0)
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None


# -------------------------------------------------------------------
# Secrets → environment hydration
# -------------------------------------------------------------------

def _hydrate_openai_env_from_secrets() -> None:
    """
    Ensure OPENAI_API_KEY and USE_OPENAI are set in os.environ if possible.

    This makes the answerer self-contained so it works even if the Streamlit
    main script didn't set env vars yet (e.g., on Streamlit Cloud).
    """
    # USE_OPENAI: default to "true" unless explicitly disabled
    if os.environ.get("USE_OPENAI") is None:
        default_val = "true"
        if st is not None:
            try:
                raw = st.secrets.get("USE_OPENAI", None)
                if raw is not None:
                    default_val = str(raw).lower()
            except Exception:
                pass
        os.environ["USE_OPENAI"] = "true" if default_val == "true" else "false"

    # If key already present, don't touch it
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return

    if st is None:
        return

    key = ""
    try:
        # Root-level candidates
        for k in ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_APIKEY", "openai_api_key", "openai_key"]:
            if k in st.secrets:
                v = str(st.secrets[k]).strip()
                if v:
                    key = v
                    break

        # Section-based
        if not key:
            for section in ("openai", "OPENAI", "llm"):
                sec = st.secrets.get(section, None)
                if isinstance(sec, dict):
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
    - USE_OPENAI=true (after hydration)
    - OPENAI_API_KEY non-empty
    - openai client import succeeded
    """
    if _OpenAI is None:
        return False

    _hydrate_openai_env_from_secrets()

    if os.environ.get("USE_OPENAI", "false").lower() != "true":
        return False

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return False

    return True


# -------------------------------------------------------------------
# Context building & extractive fallback
# -------------------------------------------------------------------

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
        # 2 chars for blank lines
        if total + len(txt) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(txt[:remaining])
            break
        parts.append(txt)
        total += len(txt) + 2
    return "\n\n".join(parts).strip()


def _extractive_fallback(recs: List[Dict], question: str, max_chars: int, reason: str, error: str = "") -> Tuple[str, Dict]:
    """
    Simple extractive answer: stitch together the most relevant chunks.
    Also records why we fell back (for debugging).
    """
    joined = "\n\n".join((r.get("text") or "") for r in recs).strip()
    if not joined:
        ans = "No relevant context found in the current corpus."
    else:
        ans = joined[:max_chars]

    meta = {
        "llm": "extractive",
        "used_openai": False,
        "reason": reason,
    }
    if error:
        meta["error"] = error
    return ans, meta


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

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
    # No context at all
    if not recs:
        return _extractive_fallback(
            recs,
            question,
            max_chars,
            reason="no_context",
        )

    # Try OpenAI if enabled
    if _openai_enabled():
        try:
            # Create client from env
            client = _OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())

            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
            # Very rough: chars ≈ 3 tokens
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
                # Unexpected empty answer → fallback
                return _extractive_fallback(
                    recs,
                    question,
                    max_chars,
                    reason="openai_empty_response",
                )

            # Hard character trim
            if len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."

            meta = {
                "llm": model,
                "used_openai": True,
                "reason": "openai_success",
            }
            return text, meta

        except Exception as e:
            # IMPORTANT: record the error so we can see it in the UI
            return _extractive_fallback(
                recs,
                question,
                max_chars,
                reason="openai_exception",
                error=str(e),
            )

    # OpenAI disabled / not available
    return _extractive_fallback(
        recs,
        question,
        max_chars,
        reason="openai_disabled_or_no_key",
    )
