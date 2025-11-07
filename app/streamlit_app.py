# app/streamlit_app.py
import os, time, requests, streamlit as st

# ---- Secrets / Env ----
def _get_secret(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)

API = _get_secret("EKA_API", "http://127.0.0.1:8000")
TIMEOUT_S = int(_get_secret("EKA_TIMEOUT_S", "25"))
TOP_K = int(_get_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_get_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _get_secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER = _get_secret("EKA_USE_RERANKER", "true").lower() == "true"

st.set_page_config(page_title="Enterprise Knowledge Agent", page_icon="✨", layout="wide")

st.markdown("""
<style>
.header-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.badge { border-radius: 999px; padding: 4px 10px; font-size: 0.85rem; border: 1px solid transparent; }
.badge-ok  { background:#E8FFF3; color:#05603A; border-color:#ABEFC6; }
.badge-warm{ background:#FFF7ED; color:#9A3412; border-color:#FED7AA; }
.badge-err { background:#FFF1F1; color:#B42318; border-color:#FECDCA; }
.kpi{padding:8px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#FCFEFF;display:inline-block;margin-right:8px;}
.small{font-size:0.92rem;color:#687076;}
.cv-legend { display:flex; gap:10px; align-items:center; margin-bottom:6px; }
.cv-dot { width:10px; height:10px; display:inline-block; border-radius:999px; border:1px solid #d1d5db; }
.cv-b1 { background:#FEE2E2; } .cv-b2 { background:#FEF9C3; } .cv-b3 { background:#ECFEFF; } .cv-b4 { background:#F0FDF4; }
.cv-pill { display:inline-block; margin:4px 6px 8px 0; padding:6px 10px; border-radius:999px; border:1px solid #e5e7eb; font-size: 0.95rem; }
.progress-wrap { display:flex; align-items:center; gap:8px; }
.bar { width:140px; height:8px; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; background:#fafafa; }
.bar > div { height:100%; background:#10b981; }
</style>
""", unsafe_allow_html=True)

def post_json(path, payload, timeout=None):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout or TIMEOUT_S)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def post_files(path, files, data=None, timeout=None):
    try:
        r = requests.post(f"{API}{path}", files=files, data=data or {}, timeout=timeout or TIMEOUT_S)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def get_json(path, timeout=None):
    try:
        r = requests.get(f"{API}{path}", timeout=timeout or 8)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def status_badge_html():
    s, e = get_json("/healthz")
    if e:
        return "<span class='badge badge-err'>Offline</span>", "err"
    if (s or {}).get("index_state") in ("queued","running"):
        return "<span class='badge badge-warm'>Loading</span>", "warm"
    return "<span class='badge badge-ok'>Online</span>", "ok"

# Header
st.markdown("<div class='header-row'><div><h3>Enterprise Knowledge Agent</h3><div class='small'>Ask in plain language. Agent plans, retrieves, answers, cites.</div></div><div id='status-slot'></div></div>", unsafe_allow_html=True)
slot = st.empty()
badge_html, _ = status_badge_html()
slot.markdown(f"<div style='display:flex; justify-content:flex-end'>{badge_html}</div>", unsafe_allow_html=True)

tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

# --- Ask ---
with tab_ask:
    st.markdown("**Finds relevant passages and answers strictly from them. Shows citations and a sentence-to-source map.**")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are values applied in performance reviews?")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    if st.button("Get answer", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            data, err = post_json("/answer_with_citations", {"question": q, "k": TOP_K, "max_chars": max_chars, "mode": RETRIEVAL_MODE, "rerank": USE_RERANKER})
            if err: st.error(err)
            else:
                st.subheader("Answer"); st.write(data.get("answer",""))
                meta = data.get("meta", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='kpi'>Sources: <b>{len(data.get('citations', []))}</b></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='kpi'>Retrieval: <b>{meta.get('mode','hybrid')}</b></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='kpi'>Latency: <b>{meta.get('retrieval_ms',0)} ms</b></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='kpi'>LLM: <b>{meta.get('llm','extractive')}</b></div>", unsafe_allow_html=True)

                def band(s): return 4 if s>=0.75 else 3 if s>=0.50 else 2 if s>=0.30 else 1
                def bcls(b): return {1:"cv-b1",2:"cv-b2",3:"cv-b3",4:"cv-b4"}[b]
                st.markdown("##### Why this answer: Context Visualizer")
                st.markdown("<div class='cv-legend small'><span>Evidence strength:</span><span class='cv-dot cv-b1'></span><span class='small'>weak</span><span class='cv-dot cv-b2'></span><span class='small'>medium</span><span class='cv-dot cv-b3'></span><span class='small'>strong</span><span class='cv-dot cv-b4'></span><span class='small'>very strong</span></div>", unsafe_allow_html=True)
                pills = []
                for row in (data.get("attribution") or []):
                    s = float(row.get("score",0.0)); sent = (row.get("sentence","") or "").strip()
                    src = row.get("best_source",""); rk = row.get("best_rank","-")
                    title = f"Top source #{rk}\n{src}\n(score {s:.2f})"
                    pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
                st.markdown("".join(pills), unsafe_allow_html=True)

                st.markdown("#### Sources")
                for c in (data.get("citations", []) or []):
                    with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                        st.write(c.get("preview",""))

# --- Agent ---
with tab_agent:
    st.markdown("**Plans the task, rewrites vague queries, retrieves evidence, drafts an answer, critiques it, and can create a wiki page or ticket for gaps.**")
    msg = st.text_area("Goal or task", height=110, placeholder="e.g., Create an example-rich note on how values influence hiring. If gaps exist, propose a wiki draft.")
    auto = st.checkbox("Allow auto-actions (create wiki/ticket, summarize uploads)", value=False)

    prog_slot = st.sidebar.container()
    prog_bar = prog_slot.progress(0, text="Idle")

    if st.button("Run agent", type="primary"):
        if not msg.strip(): st.warning("Please describe what you want the agent to do.")
        else:
            data, err = post_json("/agent_run", {"message": msg, "auto_actions": auto}, timeout=60)
            if err: st.error(err)
            else:
                st.subheader("Agent answer"); st.write(data.get("answer",""))
                st.markdown("#### Plan & Trace")
                for step in data.get("trace", []):
                    st.markdown(f"- **Step {step['step']}: {step['action']}**")
                    if step["action"] == "rewrite_query": st.code(step.get("output",""))
                    elif step["action"] == "retrieve":
                        for r in step.get("results", []): st.markdown(f"  • **{r['source']}** — {r['preview']}")
                    elif step["action"] == "critic":
                        st.write(f"  • Confidence: {step.get('confidence'):.2f}")
                        if step.get("gaps"): st.write("  • Gaps:"); [st.write(f"    - {g}") for g in step["gaps"]]
                        if step.get("actions"): st.write("  • Actions:"); st.json(step["actions"])

                st.info("If a wiki page was upserted, the index is rebuilding. Progress shown on the right.")
                start = time.time(); last_pct = -1
                while True:
                    s, e2 = get_json("/index_status", timeout=6)
                    if e2: break
                    pct = int(s.get("progress_pct", 0))
                    phase = s.get("phase","").capitalize()
                    msg2 = s.get("message","")
                    state = s.get("state","idle")
                    if pct != last_pct:
                        prog_bar.progress(max(1, pct), text=f"{phase}: {msg2}")
                        if state in ("running","queued"):
                            slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
                        last_pct = pct
                    if state in ("done","error"): break
                    if time.time() - start > 600: break
                    time.sleep(0.35)

                if s and s.get("state") == "done":
                    prog_bar.progress(100, text="Done: index updated")
                    slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
                elif s and s.get("state") == "error":
                    prog_bar.progress(pct, text="❌ Indexing failed")
                    slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-err'>Offline</span></div>", unsafe_allow_html=True)

# --- Upload ---
with tab_upload:
    st.markdown("**Adds plain-text or Markdown files to your knowledge base. They will be cited in answers.**")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & update knowledge"):
        if not f:
            st.warning("Please select a file first.")
        else:
            data, err = post_files("/ingest_file", files={"file": (f.name, f.getvalue())}, data={"title": ttl})
            if err: st.error(err)
            else:
                st.success("Upload received. Rebuilding the index in the background…")
                prog = st.progress(0, text="Queued…")
                info = st.empty()
                start = time.time(); last_pct = -1
                while True:
                    s, e2 = get_json("/index_status", timeout=6)
                    if e2: info.error(e2); time.sleep(0.5); continue
                    pct = int(s.get("progress_pct", 0))
                    phase = s.get("phase","").capitalize()
                    msg2 = s.get("message","")
                    state = s.get("state","idle")
                    if pct != last_pct:
                        prog.progress(max(1, pct), text=f"{phase}: {msg2}")
                        if state in ("running","queued"):
                            slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
                        last_pct = pct
                    if state in ("done","error"): break
                    if time.time() - start > 600:
                        info.info("Indexing still running; you can continue using the app.")
                        break
                    time.sleep(0.3)
                if s.get("state") == "done":
                    prog.progress(100, text="Done: index updated")
                    slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
                elif s.get("state") == "error":
                    prog.progress(pct, text="❌ Indexing failed")
                    slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-err'>Offline</span></div>", unsafe_allow_html=True)

# --- About ---
with tab_about:
    st.markdown("### How this works")
    st.markdown("""
- **Hybrid retrieval**: semantic + keyword search for recall and precision.
- **Reranker**: cross-encoder refines the top candidates.
- **Q&A**: answers strictly from retrieved passages; cites sources.
- **Context Visualizer**: each sentence maps to the strongest supporting passage.
- **Async indexing**: uploads return instantly; reindex runs with live progress until **Done**.
- **GPT-optional**: server-side. The Streamlit UI never reads your key.
""")
    st.info(f"Backend API: {API}")
