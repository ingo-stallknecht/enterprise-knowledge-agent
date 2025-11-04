# app/streamlit_app.py
import json
import requests
import streamlit as st

st.set_page_config(page_title="Enterprise Knowledge Agent (Local)", layout="wide")

# -----------------------
# Sidebar: connection
# -----------------------
st.sidebar.title("Connection")
API = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
timeout_s = st.sidebar.slider("Request timeout (s)", 3, 30, 10)

def get_json(url: str, method: str = "GET", **kwargs):
    """HTTP helper that never crashes the app; returns (data, error_str)."""
    try:
        if method.upper() == "GET":
            r = requests.get(url, timeout=timeout_s, **kwargs)
        else:
            r = requests.post(url, timeout=timeout_s, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def pretty_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def show_citations(citations):
    if not citations:
        st.info("No citations returned.")
        return
    for c in citations:
        with st.expander(f"Source: {c.get('source')}  (score={c.get('score'):.3f})", expanded=False):
            st.code(c.get("preview", ""), language="markdown")

st.title("Enterprise Knowledge Agent (Local)")

tab_ask, tab_agent, tab_upload, tab_health = st.tabs(["Ask", "Agent", "Upload", "Health"])

# -----------------------
# Health tab
# -----------------------
with tab_health:
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Check backend / reconnect"):
            pass

    data, err = get_json(f"{API}/healthz")
    if err:
        st.error(f"Could not reach backend: {err}")
        st.info(
            "Make sure FastAPI is running. "
            "Tip: start it without auto-reload for stability:\n\n"
            ".venv\\Scripts\\python -m uvicorn app.server:app --port 8000 --log-level info"
        )
    else:
        st.success("Backend reachable ✅")
        st.json(data)

# -----------------------
# Ask tab (Q&A with citations)
# -----------------------
with tab_ask:
    q = st.text_input("Ask a question")
    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Top K", 1, 12, 6)
    with c2:
        max_chars = st.slider("Max answer characters", 200, 2000, 800, step=50)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Answer with citations", use_container_width=True):
            payload = {"question": q, "k": k, "max_chars": max_chars}
            data, err = get_json(f"{API}/answer_with_citations", method="POST", json=payload)
            if err:
                st.error(err)
            else:
                st.subheader("Answer")
                st.write(data.get("answer", "").strip() or "_(empty)_")
                st.subheader("Citations")
                show_citations(data.get("citations", []))

    with colB:
        if st.button("Retrieve (debug view)", use_container_width=True):
            payload = {"query": q, "k": k}
            data, err = get_json(f"{API}/retrieve", method="POST", json=payload)
            if err:
                st.error(err)
            else:
                st.subheader("Raw retrieval results")
                st.code(pretty_json(data), language="json")

# -----------------------
# Agent tab (natural language)
# -----------------------
with tab_agent:
    st.write("Give the agent a natural-language instruction. Examples:")
    st.markdown(
        """
        - **Create ticket:** _Create a ticket: onboarding flow is broken, high priority_
        - **Upsert wiki:** _Add a wiki 'Onboarding Notes' with bullets: step 1, step 2_
        - **Ingest URL:** _Ingest URL https://about.gitlab.com/handbook/values/_
        - **Answer:** _How do I propose a change to the handbook?_
        """
    )
    msg = st.text_area("Instruction", height=120)
    if st.button("Send to /agent", use_container_width=True):
        payload = {"message": msg}
        data, err = get_json(f"{API}/agent", method="POST", json=payload)
        if err:
            st.error(err)
        else:
            st.subheader("Agent response")
            st.code(pretty_json(data), language="json")
            # if it returned an answer with citations, render nicely too
            if isinstance(data, dict) and data.get("citations"):
                st.subheader("Citations")
                show_citations(data["citations"])

# -----------------------
# Upload tab (file → wiki → reindex)
# -----------------------
with tab_upload:
    st.write("Upload a `.md` / `.txt` (PDFs are stored as attachments in this demo).")
    f = st.file_uploader("Choose file")
    ttl = st.text_input("Optional title for the wiki page")
    if st.button("Upload & ingest", use_container_width=True):
        if not f:
            st.warning("Please select a file first.")
        else:
            files = {"file": (f.name, f.getvalue())}
            data = {"title": ttl}
            try:
                r = requests.post(f"{API}/ingest_file", files=files, data=data, timeout=timeout_s)
                r.raise_for_status()
                resp = r.json()
                st.success("Ingested successfully ✔")
                st.code(pretty_json(resp), language="json")
                st.info("You can now ask a question in the 'Ask' tab. The index was rebuilt.")
            except Exception as e:
                st.error(str(e))
