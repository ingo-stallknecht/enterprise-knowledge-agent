# app/streamlit_mono.py
# Streamlit-only Enterprise Knowledge Agent (runs entirely inside Streamlit Cloud)

import sys, os, pathlib, re, time, tempfile
from typing import List, Dict, Tuple
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0) Writable cache sandbox for Streamlit Cloud (fixes HF PermissionError)
# ──────────────────────────────────────────────────────────────────────────────
def _secret(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)

# Put EVERYTHING in /mount/tmp (writable), or OS tmp as fallback
WRITABLE_BASE = pathlib.Path(_secret("EKA_CACHE_DIR", "/mount/tmp")).expanduser()
if not WRITABLE_BASE.exists():
    WRITABLE_BASE = pathlib.Path(tempfile.gettempdir()) / "eka"
HF_CACHE = WRITABLE_BASE / "hf"
DOTCACHE = WRITABLE_BASE / ".cache" / "huggingface"
DOTHF = WRITABLE_BASE / ".huggingface"

for p in (WRITABLE_BASE, HF_CACHE, DOTCACHE, DOTHF):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Make ~ point into WRITABLE_BASE so any "~/.cache" reads are safe
os.environ["HOME"] = str(WRITABLE_BASE)

# Absolutely disable any token search on HF Hub & all telemetry
os.environ["HF_HUB_DISABLE_TOKEN_SEARCH"] = "1"
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["HF_TOKEN"] = ""
os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_CACHE_SYMLINKS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Point every known cache var into our sandbox
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["XDG_CACHE_HOME"] = str(WRITABLE_BASE)

# Create empty token files where HF might look (~/.cache/huggingface, ~/.huggingface, and HF_CACHE)
for token_file in (HF_CACHE / "token", DOTCACHE / "token", DOTHF / "token"):
    try:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("", encoding="utf-8")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) OpenAI & app settings (via Secrets)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["USE_OPENAI"] = _secret("USE_OPENAI", "true")
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_MODEL"] = _secret("OPENAI_MODEL", "gpt-4o-mini")
os.environ["OPENAI_MAX_DAILY_USD"] = _secret("OPENAI_MAX_DAILY_USD", "0.50")

TOP_K = int(_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER_DEFAULT = _secret("EKA_USE_RERANKER", "true").lower() == "true"
DISABLE_RERANKER_BOOT = _secret("EKA_DISABLE_RERANKER", "false").lower() == "true"

# ──────────────────────────────────────────────────────────────────────────────
# 2) Make sure "app" package is importable
# ──────────────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.utils import ensure_dirs, load_cfg
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.rag.chunker import split_markdown
from app.llm.answerer import generate_answer

# ──────────────────────────────────────────────────────────────────────────────
# 3) Config
# ──────────────────────────────────────────────────────────────────────────────
CFG = {}
cfg_path = pathlib.Path("configs/settings.yaml")
if cfg_path.exists():
    CFG = load_cfg(str(cfg_path))

RET = CFG.get("retrieval", {})
EMB_MODEL = RET.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
NORMALIZE = bool(RET.get("normalize", True))
ALPHA = float(RET.get("hybrid_alpha", 0.6))
INDEX_PATH = RET.get("faiss_index", "data/index/handbook.index")
STORE_PATH = RET.get("store_json", "data/index/docstore.json")
CHUNK_CFG = CFG.get("chunk", {"max_chars": 1200, "overlap": 150})
ensure_dirs()

# ──────────────────────────────────────────────────────────────────────────────
# 4) UI basics
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EKA (Streamlit-only)", page_icon="✨", layout="wide")
st.markdown("""
<style>
.header-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.badge { border-radius:999px; padding:4px 10px; font-size:0.85rem; border:1px solid transparent; }
.badge-ok  { background:#E8FFF3; color:#05603A; border-color:#ABEFC6; }
.kpi{padding:8px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#FCFEFF;display:inline-block;margin-right:8px;}
.small{font-size:0.92rem;color:#687076;}
.cv-pill { display:inline-block; margin:4px 6px 8px 0; padding:6px 10px; border-radius:999px;
           border:1px solid #e5e7eb; font-size:0.95rem; }
.cv-b1 { background:#FEE2E2; } .cv-b2 { background:#FEF9C3; }
.cv-b3 { background:#ECFEFF; } .cv-b4 { background:#F0FDF4; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Cached resources (models & index)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    # Use a writable cache folder and never use a token
    emb = Embedder(EMB_MODEL, NORMALIZE)  # Embedder internally pins cache & disables token
    rer = None
    if not DISABLE_RERANKER_BOOT:
        try:
            rer = Reranker(CFG.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception:
            rer = None
    return emb, rer

@st.cache_resource(show_spinner=False)
def load_or_init_index():
    return DocIndex(INDEX_PATH, STORE_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Core helpers
# ──────────────────────────────────────────────────────────────────────────────
def _scan_processed_files() -> List[pathlib.Path]:
    return sorted(pathlib.Path("data/processed").glob("**/*.md"))

def _split_md(text: str) -> List[Dict]:
    return list(split_markdown(text, **CHUNK_CFG))

def rebuild_index(progress) -> Dict:
    files = _scan_processed_files()
    progress.update(label=f"Scanning… {len(files)} files", value=0)
    records = []
    done = 0
    for i, fp in enumerate(files, 1):
        t = fp.read_text(encoding="utf-8")
        for ch in _split_md(t):
            records.append({"text": ch["text"], "source": str(fp).replace("\\","/")})
        done = i
        progress.update(label=f"Chunking… {done}/{len(files)} files",
                        value=min(0.25, 0.05 + 0.20*(done/max(1,len(files)))))

    import numpy as np
    emb, _ = get_models()
    texts = [r["text"] for r in records]
    if texts:
        vecs = []
        batch = 64
        for s in range(0, len(texts), batch):
            e = min(len(texts), s+batch)
            vecs.append(emb.encode(texts[s:e]))
            progress.update(label=f"Embedding… {e}/{len(texts)} chunks",
                            value=0.25 + 0.60*(e/max(1,len(texts))))
        X = np.vstack(vecs)
    else:
        X = np.zeros((0, 384), dtype="float32")

    idx = load_or_init_index()
    progress.update(label="Writing index…", value=0.9)
    idx.build(X, records)
    progress.update(label="Done", value=1.0)
    return {"num_files": len(files), "num_chunks": len(records)}

def retrieve(query: str, k: int, mode: str, use_reranker: bool):
    t0 = time.time()
    emb, rer = get_models()
    qv = emb.encode([query])
    idx = load_or_init_index()
    if mode == "dense":
        raw = idx.query_dense(qv, k=max(k, 30)); meta_mode = "dense"
    elif mode == "sparse":
        raw = idx.query_sparse(query, k=max(k, 30)); meta_mode = "sparse"
    else:
        raw = idx.query_hybrid(qv, query, k=max(k, 30), alpha=ALPHA); meta_mode = "hybrid"
    candidates = [r for r,_ in raw]
    reranked = raw
    rerank_used = False
    if use_reranker and rer is not None and candidates:
        pairs = rer.rerank(query, candidates, top_k=max(k,1))
        reranked = [(rec, sc) for rec, sc in pairs]
        rerank_used = True
    top_records = [r for r,_ in reranked[:k]]
    meta = {"mode": meta_mode, "alpha": ALPHA, "rerank_used": rerank_used,
            "retrieval_ms": int((time.time()-t0)*1000)}
    return top_records, reranked, meta

# ──────────────────────────────────────────────────────────────────────────────
# 7) Context Visualizer
# ──────────────────────────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')
def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = _SENT_SPLIT.split(t)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 35 and buf:
            buf = f"{buf} {p}"
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def attribute(answer: str, records: List[Dict]) -> List[dict]:
    if not answer or not records: return []
    emb, _ = get_models()
    sents = _split_sentences(answer)
    if not sents: return []
    svecs = emb.encode(sents)
    rvecs = emb.encode([(r.get("text") or "") for r in records])
    import numpy as np
    sim = (svecs @ rvecs.T).astype(float)
    out = []
    for i, s in enumerate(sents):
        j = int(np.argmax(sim[i]))
        best = records[j]
        out.append({
            "sentence": s,
            "best_source": best.get("source"),
            "best_rank": j+1,
            "score": float(sim[i,j]),
            "preview": (best.get("text") or "")[:240] +
                       ("…" if len((best.get("text") or ""))>240 else "")
        })
    return out

def fmt_citations(pairs: List[Tuple[Dict,float]], k: int) -> List[dict]:
    cites = []
    for i,(r,sc) in enumerate(pairs[:k],1):
        txt = (r.get("text") or "").strip()
        cites.append({
            "rank": i,
            "source": r.get("source"),
            "score": float(sc),
            "preview": (txt[:280]+"…") if len(txt)>280 else txt
        })
    return cites

# ──────────────────────────────────────────────────────────────────────────────
# 8) UI with Ask / Agent / Upload / About
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='header-row'><div><h3>Enterprise Knowledge Agent (Streamlit-only)</h3>"
    "<div class='small'>Runs fully inside Streamlit. Secrets-powered OpenAI. No external backend.</div></div>"
    "<div><span class='badge badge-ok'>Online</span></div></div>",
    unsafe_allow_html=True
)

tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

# --- Ask ---
with tab_ask:
    st.markdown("**Answers from your local corpus (GitLab + uploads). Shows citations and sentence→source mapping.**")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are values applied in performance reviews?")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    colA, colB = st.columns(2)
    mode = colA.selectbox("Retrieval mode", ["hybrid","dense","sparse"],
                          index=["hybrid","dense","sparse"].index(RETRIEVAL_MODE))
    use_rr = colB.checkbox("Use reranker", value=USE_RERANKER_DEFAULT)
    if st.button("Get answer", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)
            ans, llm_meta = generate_answer(recs, q, max_chars=max_chars)
            st.subheader("Answer")
            st.write(ans)
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='kpi'>Sources: <b>{len(pairs[:TOP_K])}</b></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='kpi'>Retrieval: <b>{meta.get('mode','')}</b></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='kpi'>Latency: <b>{meta.get('retrieval_ms',0)} ms</b></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='kpi'>LLM: <b>{llm_meta.get('llm','extractive')}</b></div>", unsafe_allow_html=True)

            st.markdown("##### Why this answer: Context Visualizer")
            def band(s): return 4 if s>=0.75 else 3 if s>=0.50 else 2 if s>=0.30 else 1
            def bcls(b): return {1:"cv-b1",2:"cv-b2",3:"cv-b3",4:"cv-b4"}[b]
            pills = []
            for row in attribute(ans, recs):
                s = float(row.get("score",0))
                sent = (row.get("sentence","") or "").strip()
                src = row.get("best_source","")
                rk = row.get("best_rank","-")
                title = f"Top source #{rk}\\n{src}\\n(score {s:.2f})"
                pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
            st.markdown("".join(pills), unsafe_allow_html=True)

            st.markdown("#### Sources")
            for c in fmt_citations(pairs, TOP_K):
                with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}"):
                    st.write(c['preview'])

# --- Agent ---
def agent_run(message: str, auto_actions: bool = False) -> Dict:
    trace: List[Dict] = []
    plan = ["rewrite_query", "retrieve", "draft", "critic"]
    trace.append({"step": 1, "action": "plan", "output": plan})

    query = (message or "").strip()
    if len(query) < 12 or not query.endswith("?"):
        query = (query.rstrip(".") + "?")
    trace.append({"step": 2, "action": "rewrite_query", "output": query})

    records, pairs, _meta = retrieve(query, TOP_K, "hybrid", use_reranker=False)
    trace.append({"step": 3, "action": "retrieve", "results": [
        {"source": r.get("source"), "preview": (r.get("text","")[:200] + "…")} for r,_ in pairs[:6]
    ]})

    answer, _ = generate_answer(records, query, max_chars=900)
    trace.append({"step": 4, "action": "draft", "output": (answer[:1000] + ("…" if len(answer)>1000 else ""))})

    gaps, actions = [], []
    confidence = 0.65 if len(answer) > 200 else 0.45
    if "example" not in answer.lower():
        gaps.append("Add concrete, example-driven guidance.")
        actions.append({
            "type": "upsert_wiki_draft",
            "title": "Example-Rich Guide",
            "content": "# Example-Rich Guide\n\nAdd concrete Q&A and scenarios mapped to values.\n\n- Example 1 …\n- Example 2 …\n"
        })

    applied = []
    if auto_actions and actions:
        wiki_dir = pathlib.Path("data/processed/wiki"); wiki_dir.mkdir(parents=True, exist_ok=True)
        for a in actions:
            if a.get("type") == "upsert_wiki_draft":
                slug = re.sub(r"[^a-z0-9\-]+", "-", a["title"].lower()).strip("-") or "page"
                (wiki_dir / f"{slug}.md").write_text(a["content"], encoding="utf-8")
                applied.append({"upserted": f"{slug}.md"})
        if applied:
            prog = st.progress(0.0, text="Agent: updating index…")
            rebuild_index(prog)

    step5 = {"step": 5, "action": "critic", "confidence": confidence}
    if gaps: step5["gaps"] = gaps
    if actions: step5["actions"] = actions
    trace.append(step5)

    return {"answer": answer, "trace": trace, "applied_actions": applied}

with tab_agent:
    st.markdown("**Agent: plan → rewrite → retrieve → draft → critic. Can upsert a wiki draft and reindex.**")
    msg = st.text_area("Goal or task", height=110,
                       placeholder="e.g., Create an example-rich note on values in performance reviews; if gaps, propose a wiki draft.")
    auto = st.checkbox("Allow auto-actions (create wiki draft + reindex)", value=False)
    if st.button("Run agent", type="primary"):
        if not msg.strip():
            st.warning("Please describe what you want the agent to do.")
        else:
            res = agent_run(msg, auto_actions=auto)
            st.subheader("Agent answer"); st.write(res.get("answer",""))
            st.markdown("#### Plan & Trace")
            for step in res.get("trace", []):
                st.markdown(f"- **Step {step['step']}: {step['action']}**")
                if step["action"] == "rewrite_query":
                    st.code(step.get("output",""))
                elif step["action"] == "retrieve":
                    for r in step.get("results", []): st.markdown(f"  • **{r['source']}** — {r['preview']}")
                elif step["action"] == "critic":
                    st.write(f"  • Confidence: {step.get('confidence',0.0):.2f}")
                    if step.get("gaps"): st.write("  • Gaps:"); [st.write(f"    - {g}") for g in step["gaps"]]
                    if step.get("actions"): st.write("  • Proposed actions:"); st.json(step["actions"])
            if res.get("applied_actions"):
                st.success(f"✅ Applied: {res['applied_actions']}")

# --- Upload ---
with tab_upload:
    st.markdown("**Upload `.md` or `.txt` — added to the knowledge base.**")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & rebuild index"):
        if not f:
            st.warning("Please select a file first.")
        else:
            name = ttl.strip() or f.name
            slug = re.sub(r"[^a-z0-9\\-]+", "-", name.lower()).strip("-") or "page"
            wiki_dir = pathlib.Path("data/processed/wiki"); wiki_dir.mkdir(parents=True, exist_ok=True)
            text = f.getvalue().decode("utf-8", errors="ignore")
            (wiki_dir / f"{slug}.md").write_text(text, encoding="utf-8")
            prog = st.progress(0.0, text="Queued…")
            stats = rebuild_index(prog)
            st.success(f"✅ Indexed {stats['num_chunks']} chunks from {stats['num_files']} files.")

# --- About ---
with tab_about:
    st.markdown("""
### How this works (Streamlit-only)
- **Hybrid retrieval** (FAISS dense + TF-IDF) with optional Cross-Encoder reranker  
- **Q&A:** extractive by default; GPT-assisted if `USE_OPENAI=true`  
- **Agent:** short chain (plan→rewrite→retrieve→draft→critic) with optional wiki upsert + reindex  
- **Uploads:** `.md`/`.txt` stored under `data/processed/wiki`; rebuild index on demand  
- **Caching:** models cached via `st.cache_resource` while app stays warm  
- **Note:** Streamlit Community Cloud storage is ephemeral — keep core corpus in Git or fetch on first run.
""")
