# app/streamlit_app.py
import os
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ----------------- App config -----------------
load_dotenv()
API = os.environ.get("EKA_API") or "http://127.0.0.1:8000"
TIMEOUT_S = 25
TOP_K = 12
MAX_CHARS_DEFAULT = 900
RETRIEVAL_MODE = "hybrid"
USE_RERANKER = True

st.set_page_config(page_title="Enterprise Knowledge Agent", page_icon="✨", layout="wide")

# ----------------- Styles (incl. CV bubbles) -----------------
st.markdown("""
<style>
.kpi{padding:8px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#FCFEFF;display:inline-block;margin-right:8px;}
.small{font-size:0.92rem;color:#687076;}
.cv-legend { display:flex; gap:10px; align-items:center; margin-bottom:6px; }
.cv-dot { width:10px; height:10px; display:inline-block; border-radius:999px; border:1px solid #d1d5db; }
.cv-b1 { background:#FEE2E2; } .cv-b2 { background:#FEF9C3; } .cv-b3 { background:#ECFEFF; } .cv-b4 { background:#F0FDF4; }
.cv-pill { display:inline-block; margin:6px 6px 10px 0; padding:8px 10px; border-radius:999px; border:1px solid #e5e7eb; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ----------------- Floating status (no flicker when done) -----------------
STATUS_WIDGET = f"""
<style>
#eka-status {{
  position: fixed; top: 10px; right: 12px; z-index: 1000;
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 999px;
  padding: 8px 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.06);
  font-size: 0.92rem; display: flex; gap: 10px; align-items: center;
  pointer-events: none;   /* don't block clicks */
}}
#eka-status .dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
.dot-ok {{ background: #10b981; }}
.dot-warm {{ background: #f59e0b; }}
.dot-err {{ background: #ef4444; }}
#eka-status .phase {{
  padding: 2px 8px; border-radius: 999px; border: 1px solid #e5e7eb;
  background: #f9fafb; font-size: 0.85rem;
}}
#eka-status .bar {{
  position: relative; width: 160px; height: 8px; border-radius: 999px;
  background: #eef2f7; overflow: hidden; display:none;
}}
#eka-status .bar > div {{
  height: 100%; border-radius: 999px; background: #60a5fa; width: 0%;
  transition: width 200ms linear;
}}
</style>
<div id="eka-status">
  <span class="dot dot-err" id="eka-dot"></span>
  <span id="eka-text">Offline</span>
  <span class="phase" id="eka-phase" style="display:none">Idle</span>
  <div class="bar" id="eka-bar"><div id="eka-bar-inner"></div></div>
</div>
<script>
const API = "{API}";
const dot   = document.getElementById('eka-dot');
const txt   = document.getElementById('eka-text');
const phase = document.getElementById('eka-phase');
const bar   = document.getElementById('eka-bar');
const fill  = document.getElementById('eka-bar-inner');

function showBar(pct, label) {{
  bar.style.display = "block";
  fill.style.width = Math.max(0, Math.min(100, parseInt(pct||0))) + "%";
  phase.style.display = "inline-block";
  phase.textContent = label;
}}

function hideBar(label) {{
  bar.style.display = "none";
  fill.style.width = "0%";
  phase.style.display = "inline-block";
  phase.textContent = label;
}}

async function poll() {{
  try {{
    const hz = await fetch(API + "/healthz", {{cache: "no-store"}});
    if (!hz.ok) throw new Error("healthz " + hz.status);
    const st = await fetch(API + "/index_status", {{cache: "no-store"}});
    const s = await st.json();

    dot.className = "dot dot-ok";
    txt.textContent = "Online";

    const state = (s.state || "idle").toLowerCase();
    const ph = (s.phase || "idle");
    const pct = Math.max(0, Math.min(100, parseInt(s.progress_pct || 0)));
    const warming = ["queued","running","finalizing"].includes(state);

    if (warming) {{
      dot.className = "dot dot-warm";
      txt.textContent = "Warming up";
      showBar(pct, ph.charAt(0).toUpperCase() + ph.slice(1));
      localStorage.setItem("eka_prev_state", "running");
    }} else if (state === "done") {{
      // Immediately hide bar; no flicker
      hideBar("Done");
      const prev = localStorage.getItem("eka_prev_state") || "";
      if (prev === "running") {{
        const evt = new CustomEvent("eka_index_done");
        window.dispatchEvent(evt);
      }}
      localStorage.setItem("eka_prev_state", "done");
    }} else {{
      hideBar(ph.charAt(0).toUpperCase() + ph.slice(1));
      localStorage.setItem("eka_prev_state", state);
    }}
  }} catch (e) {{
    dot.className = "dot dot-err";
    txt.textContent = "Offline";
    hideBar("Idle");
  }}
}}
poll();
setInterval(poll, 1200);
</script>
"""
components.html(STATUS_WIDGET, height=80, scrolling=False)

# ----------------- HTTP helpers -----------------
def post_json(path, payload, timeout=None):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout or TIMEOUT_S)
        if r.status_code == 404:
            return None, "404"
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

def get_json_with_params(path, params, timeout=None):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=timeout or TIMEOUT_S)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ----------------- Optional toast when indexing finishes -----------------
if "eka_last_state" not in st.session_state:
    st.session_state.eka_last_state = "idle"
status_now, _ = get_json("/index_status")
if status_now:
    cur = (status_now.get("state") or "idle").lower()
    prev = st.session_state.eka_last_state
    if prev in ("queued","running","finalizing") and cur == "done":
        st.toast("✅ Finished indexing — new content is now searchable.", icon="✅")
    st.session_state.eka_last_state = cur

# ----------------- Header -----------------
st.markdown("### Enterprise Knowledge Agent")
st.caption("Ask in plain language. The agent plans, retrieves, answers with citations, and keeps indexing progress visible across tabs.")

st.write("")
tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

# ----------------- Ask tab -----------------
with tab_ask:
    st.markdown("**What it does:** Finds the most relevant passages and answers strictly from them. Shows citations and a sentence-to-source map.")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are company values applied in performance reviews?")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    if st.button("Get answer", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            data, err = post_json(
                "/answer_with_citations",
                {"question": q, "k": TOP_K, "max_chars": max_chars, "mode": RETRIEVAL_MODE, "rerank": USE_RERANKER}
            )
            if err:
                st.error(err)
            else:
                st.subheader("Answer")
                st.write(data.get("answer",""))
                meta = data.get("meta", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='kpi'>Sources: <b>{len(data.get('citations', []))}</b></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='kpi'>Retrieval: <b>{meta.get('mode','hybrid')}</b></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='kpi'>Latency: <b>{meta.get('retrieval_ms',0)} ms</b></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='kpi'>LLM: <b>{meta.get('llm','extractive')}</b></div>", unsafe_allow_html=True)

                # Context Visualizer (colored bubbles)
                def band(s): return 4 if s>=0.75 else 3 if s>=0.50 else 2 if s>=0.30 else 1
                def bcls(b): return {1:"cv-b1",2:"cv-b2",3:"cv-b3",4:"cv-b4"}[b]
                st.markdown("##### Why this answer: Context Visualizer")
                st.markdown("<div class='cv-legend small'><span>Evidence strength:</span><span class='cv-dot cv-b1'></span><span class='small'>weak</span><span class='cv-dot cv-b2'></span><span class='small'>medium</span><span class='cv-dot cv-b3'></span><span class='small'>strong</span><span class='cv-dot cv-b4'></span><span class='small'>very strong</span></div>", unsafe_allow_html=True)
                pills = []
                for row in (data.get("attribution") or []):
                    s = float(row.get("score",0.0)); sent = (row.get("sentence","") or "").strip()
                    src = row.get("best_source",""); rk = row.get("best_rank","-")
                    title = f"Top source #{rk}\\n{src}\\n(score {s:.2f})"
                    pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
                st.markdown("".join(pills), unsafe_allow_html=True)

                st.markdown("#### Sources")
                for c in (data.get("citations", []) or []):
                    with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                        st.write(c.get("preview",""))

# --- Agent tab (clean, human-readable) ---
with tab_agent:
    st.markdown("**What it does:** Plans the task, rewrites vague queries, retrieves evidence, drafts an answer, critiques it, and can create a wiki page or ticket for gaps.")
    msg = st.text_area("Goal or task", height=110, placeholder="e.g., Create an example-rich note on how values influence hiring. If gaps exist, propose a wiki draft.")
    auto = st.checkbox("Allow auto-actions (create wiki/ticket, summarize uploads)", value=False)
    if st.button("Run agent", type="primary"):
        if not msg.strip():
            st.warning("Please describe what you want the agent to do.")
        else:
            data, err = post_json("/agent_run", {"message": msg, "auto_actions": auto}, timeout=60)
            if err == "404":
                data, err = get_json_with_params("/agent_run", {"message": msg, "auto_actions": str(auto).lower()}, timeout=60)
            if err:
                st.error(err)
            else:
                st.subheader("Agent answer")
                st.write(data.get("answer",""))

                st.markdown("#### Plan & Trace")
                for step in data.get("trace", []):
                    action = step.get("action","")
                    st.markdown(f"- **Step {step.get('step','?')}: {action}**")
                    if action == "rewrite_query":
                        st.markdown(f"> {step.get('output','')}")
                    elif action == "retrieve":
                        results = step.get("results", [])
                        if results:
                            for r in results:
                                st.markdown(f"  • **{r.get('source','')}** — {r.get('preview','')}")
                        else:
                            st.markdown("  • (no results)")
                    elif action == "draft":
                        st.markdown(f"> {step.get('output','')}")
                    elif action == "critic":
                        st.markdown(f"  • Confidence: **{step.get('confidence', step.get('Confidence', 0)):.2f}**")
                        gaps = step.get("gaps") or step.get("Gaps") or []
                        if gaps:
                            st.markdown("  • Gaps:")
                            for g in gaps:
                                st.markdown(f"    - {g}")
                        actions = step.get("actions") or []
                        if actions:
                            st.markdown("  • Proposed actions:")
                            for a in actions:
                                if a.get("type") == "upsert_wiki_draft":
                                    st.markdown(f"    - **Create wiki draft:** *{a.get('title','') }*")
                                else:
                                    st.markdown(f"    - {a}")

                # Performed actions summary (if auto_actions True)
                pa = data.get("performed_actions", [])
                if pa:
                    st.success("Actions performed:")
                    for a in pa:
                        if "error" in a:
                            st.error(a)
                            continue
                        line = f"• {a.get('type')} → {a.get('title','')}"
                        if a.get("path"):
                            line += f"  \n  `{a['path']}`"
                        st.markdown(line)

# ----------------- Upload tab -----------------
with tab_upload:
    st.markdown("**What it does:** Adds plain-text or Markdown files to your local knowledge base. The agent will cite them in answers.")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & update knowledge"):
        if not f:
            st.warning("Please select a file first.")
        else:
            try:
                r = requests.post(
                    f"{API}/ingest_file",
                    files={"file": (f.name, f.getvalue())},
                    data={"title": ttl},
                    timeout=TIMEOUT_S
                )
                r.raise_for_status()
                _ = r.json()
                st.success("Upload received. Rebuilding the index in the background…")
            except Exception as e:
                st.error(str(e))

# ----------------- About tab -----------------
with tab_about:
    st.markdown("### How this works")
    st.markdown("""
- **Hybrid retrieval**: combines semantic and keyword search for both recall and precision.  
- **Reranker**: uses a cross-encoder model to refine the top candidate passages.  
- **Q&A generation**: the LLM drafts concise, evidence-based answers with citations.  
- **Context Visualizer**: shows how each sentence maps to its strongest source (colored bubbles).  
- **Agent workflow**: plans the task, rewrites vague queries, retrieves evidence, drafts an answer, critiques it, and can create wiki pages or tickets for gaps.  
- **Async indexing**: uploads and agent upserts rebuild the index in the background, with visible progress (top-right) and a toast when finished.  
""")
