# app/llm/answerer.py

from typing import List, Dict, Tuple
import os
import textwrap
import requests

# Optional Streamlit import (for secrets on Streamlit Cloud)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None


# -------------------------------------------------------------------
# Helpers: pull config from env + secrets (no magic)
# -------------------------------------------------------------------


def _get_use_openai_flag() -> bool:
    """
    Decide whether to attempt OpenAI usage.

    Precedence:
    1) Environment variable USE_OPENAI
    2) Streamlit secrets USE_OPENAI
    3) Default: true
    """
    # 1) env
    env_flag = os.environ.get("USE_OPENAI")
    if env_flag is not None:
        return str(env_flag).lower() == "true"

    # 2) secrets
    if st is not None:
        try:
            if "USE_OPENAI" in st.secrets:
                sec_flag = str(st.secrets["USE_OPENAI"]).lower()
                return sec_flag == "true"
        except Exception:
            pass

    # 3) default
    return True


def _get_openai_key() -> str:
    """
    Read OPENAI_API_KEY from env or Streamlit secrets.
    """
    # 1) env
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    # 2) secrets
    if st is not None:
        try:
            # root-level key
            if "OPENAI_API_KEY" in st.secrets:
                key = str(st.secrets["OPENAI_API_KEY"]).strip()
                if key:
                    return key

            # section-based, e.g. [openai] API_KEY="..."
            for section in ("openai", "OPENAI", "llm"):
                try:
                    sec = st.secrets[section]  # type: ignore[index]
                except Exception:
                    sec = None
                if isinstance(sec, dict):
                    for k in ("api_key", "API_KEY", "key"):
                        v = str(sec.get(k, "")).strip()
                        if v:
                            return v
        except Exception:
            pass

    return ""


def _get_openai_model() -> str:
    """
    Determine which model to use.
    """
    # env
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if model:
        return model

    # secrets
    if st is not None:
        try:
            if "OPENAI_MODEL" in st.secrets:
                m = str(st.secrets["OPENAI_MODEL"]).strip()
                if m:
                    return m
        except Exception:
            pass

    # default
    return "gpt-4o-mini"


def _get_openai_temperature() -> float:
    # env
    env_val = os.environ.get("OPENAI_TEMPERATURE")
    if env_val is not None:
        try:
            return float(env_val)
        except Exception:
            pass

    # secrets
    if st is not None:
        try:
            if "OPENAI_TEMPERATURE" in st.secrets:
                sec_val = str(st.secrets["OPENAI_TEMPERATURE"])
                return float(sec_val)
        except Exception:
            pass

    return 0.2


def _get_max_output_tokens(default_chars: int) -> int:
    # env
    env_val = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
    if env_val is not None:
        try:
            return int(env_val)
        except Exception:
            pass

    # secrets
    if st is not None:
        try:
            if "OPENAI_MAX_OUTPUT_TOKENS" in st.secrets:
                sec_val = str(st.secrets["OPENAI_MAX_OUTPUT_TOKENS"])
                return int(sec_val)
        except Exception:
            pass

    # fallback: rough chars→tokens heuristic
    return max(200, min(900, default_chars // 3))


# -------------------------------------------------------------------
# Context building & extractive fallback
# -------------------------------------------------------------------


def _build_context(recs: List[Dict], max_chars: int = 6000) -> str:
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


def _extractive_fallback(
    recs: List[Dict],
    question: str,
    max_chars: int,
    reason: str,
    extra_meta: Dict[str, object] | None = None,
) -> Tuple[str, Dict]:
    joined = "\n\n".join((r.get("text") or "") for r in recs).strip()
    if not joined:
        ans = "No relevant context found in the current corpus."
    else:
        ans = joined[:max_chars]

    meta: Dict[str, object] = {
        "llm": "extractive",
        "used_openai": False,
        "reason": reason,
    }
    if extra_meta:
        meta.update(extra_meta)
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
    """
    if not recs:
        return _extractive_fallback(
            recs,
            question,
            max_chars,
            reason="no_context",
            extra_meta={
                "use_flag": _get_use_openai_flag(),
                "has_key_env": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                "has_key_secret": bool(st and "OPENAI_API_KEY" in getattr(st, "secrets", {})),  # type: ignore[arg-type]
            },
        )

    use_flag = _get_use_openai_flag()
    key = _get_openai_key()
    has_key_env = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_key_secret = bool(key)

    if not use_flag or not key:
        return _extractive_fallback(
            recs,
            question,
            max_chars,
            reason="openai_disabled_or_no_key",
            extra_meta={
                "use_flag": use_flag,
                "has_key_env": has_key_env,
                "has_key_secret": has_key_secret,
            },
        )

    model = _get_openai_model()
    temperature = _get_openai_temperature()
    max_tokens = _get_max_output_tokens(max_chars)

    try:
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
            - If something is not covered, explicitly say that the current corpus does not contain it.
            - Keep the answer under approximately {max_chars} characters.
            """
        ).strip()

        payload = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return _extractive_fallback(
                recs,
                question,
                max_chars,
                reason="openai_empty_choices",
                extra_meta={
                    "use_flag": use_flag,
                    "has_key_env": has_key_env,
                    "has_key_secret": has_key_secret,
                    "raw": str(data)[:400],
                },
            )

        message = choices[0].get("message", {}) or {}
        text = (message.get("content") or "").strip()
        if not text:
            return _extractive_fallback(
                recs,
                question,
                max_chars,
                reason="openai_empty_message",
                extra_meta={
                    "use_flag": use_flag,
                    "has_key_env": has_key_env,
                    "has_key_secret": has_key_secret,
                    "raw": str(data)[:400],
                },
            )

        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        meta: Dict[str, object] = {
            "llm": model,
            "used_openai": True,
            "reason": "openai_success",
            "use_flag": use_flag,
            "has_key_env": has_key_env,
            "has_key_secret": has_key_secret,
        }
        return text, meta

    except Exception as e:
        return _extractive_fallback(
            recs,
            question,
            max_chars,
            reason="openai_exception",
            extra_meta={
                "use_flag": use_flag,
                "has_key_env": has_key_env,
                "has_key_secret": has_key_secret,
                "error": str(e),
            },
        )
