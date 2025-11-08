# app/streamlit_mono.py
# Streamlit-only Enterprise Knowledge Agent (runs entirely inside Streamlit Cloud)
# Offline-safe: if HTTP bootstrap fails, we fall back to a small built-in seed corpus.

import sys, os, pathlib, re, time, tempfile
from typing import List, Dict, Tuple, Optional
import streamlit as st

# Optional lightweight HTML→Markdown
try:
    from markdownify import markdownify as md
except Exception:
    md = None

# Optional fetch
try:
    import requests
except Exception:
    requests = None


def _secret(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)


# --- Writable caches for HF & transformers (Streamlit Cloud safe) ---
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

os.environ["HOME"] = str(WRITABLE_BASE)
os.environ["HF_HUB_DISABLE_TOKEN_SEARCH"] = "1"
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["HF_TOKEN"] = ""
os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_CACHE_SYMLINKS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["XDG_CACHE_HOME"] = str(WRITABLE_BASE)
for token_file in (HF_CACHE / "token", DOTCACHE / "token", DOTHF / "token"):
    try:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("", encoding="utf-8")
    except Exception:
        pass

# --- OpenAI / LLM toggles (robust secret loading) ---
def _find_secret_key() -> str:
    candidates = [
        "OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_APIKEY",
        "openai_api_key", "openai_key",
    ]
    try:
        for k in candidates:
            if k in st.secrets:
                v = str(st.secrets[k]).strip()
                if v:
                    return v
        for section in ("openai", "OPENAI", "llm"):
            if section in st.secrets and isinstance(st.secrets[section], dict):
                for k in ("api_key", "API_KEY", "key"):
                    v = str(st.secrets[section].get(k, "")).strip()
                    if v:
                        return v
    except Exception:
        pass
    for k in candidates:
        v = str(os.getenv(k, "")).strip()
        if v:
            return v
    return ""

use_openai_flag = str(st.secrets.get("USE_OPENAI", os.environ.get("USE_OPENAI", "true"))).lower() == "true"
os.environ["USE_OPENAI"] = "true" if use_openai_flag else "false"

_openai_key = _find_secret_key()
if _openai_key:
    os.environ["OPENAI_API_KEY"] = _openai_key

os.environ["OPENAI_MODEL"] = str(st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
os.environ["OPENAI_MAX_DAILY_USD"] = str(st.secrets.get("OPENAI_MAX_DAILY_USD", os.environ.get("OPENAI_MAX_DAILY_USD", "0.50")))

def _openai_diag() -> str:
    use = os.environ.get("USE_OPENAI", "false").lower() == "true"
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "")
    if use and key:
        return f"OpenAI: ON (key ✓, model {model})"
    if use and not key:
        return "OpenAI: ON (key missing ⚠)"
    return "OpenAI: OFF"

TOP_K = int(_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER_DEFAULT = _secret("EKA_USE_RERANKER", "true").lower() == "true"
DISABLE_RERANKER_BOOT = _secret("EKA_DISABLE_RERANKER", "false").lower() == "true"
AUTO_BOOTSTRAP = _secret("EKA_BOOTSTRAP", "true").lower() == "true"

# Ensure the repo root (one level up) is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from app.rag.utils import ensure_dirs, load_cfg
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.rag.chunker import split_markdown
from app.llm.answerer import generate_answer

# --- Config & dirs ---
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
ensure_dirs()  # also creates data/raw and data/processed/wiki

RAW_DIR = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")
WIKI_DIR = PROC_DIR / "wiki"

# ---------------- Progress wrapper (fixes Streamlit 1.38 API) ----------------
class Prog:
    def __init__(self, initial_text: str = "Working…"):
        self._pb = st.progress(0, text=initial_text)
        self._last = 0.0
    def update(self, label: Optional[str] = None, value: Optional[float] = None):
        if value is None:
            value = self._last
        value = max(0.0, min(float(value), 1.0))
        self._last = value
        if label is None:
            self._pb.progress(int(value * 100))
        else:
            self._pb.progress(int(value * 100), text=label)

# ---------------- Seed corpus (offline) ----------------
SEED_MD: Dict[str, str] = {
    "values.md": """# Values at Our Company

We emphasize Collaboration, Results, Efficiency, Diversity & Belonging, Iteration, and Transparency.
Values are referenced in hiring, onboarding, and performance reviews.

## Values in performance reviews
- Reviewers cite concrete behaviors that demonstrate values.
- Examples: default to open communication; iterate in small steps; measure outcomes.
- Employees are encouraged to give evidence (issues, MRs, docs) that illustrate the values in action.
""",
    "communication.md": """# Communication

Default to asynchronous, documented communication. Prefer issues and documents to meetings. Summaries and decisions
are captured in writing. For sensitive topics, use appropriate private channels, then write a public summary when possible.
""",
    "engineering-management.md": """# Engineering Management

Managers coach iteration (ship small), measurable outcomes, and transparent decision logs. One-on-ones focus on growth,
feedback, and removing blockers. Managers role-model values and cite them in feedback.
""",
}

def write_seed_corpus() -> Dict:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in SEED_MD.items():
        (WIKI_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")
    (PROC_DIR / "values.md").write_text(SEED_MD["values.md"], encoding="utf-8")
    (PROC_DIR / "communication.md").write_text(SEED_MD["communication.md"], encoding="utf-8")
    return {"ok": True, "written": len(SEED_MD) + 2}

# ---------------- Bootstrap helpers ----------------
CURATED = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/handbook/values/",
    "https://about.gitlab.com/handbook/engineering/",
    "https://about.gitlab.com/handbook/people-group/",
    "https://about.gitlab.com/handbook/communication/",
    "https://about.gitlab.com/handbook/product/",
    "https://about.gitlab.com/handbook/sales/",
    "https://about.gitlab.com/handbook/marketing/",
    "https://about.gitlab.com/handbook/leadership/",
    "https://about.gitlab.com/handbook/engineering/management/",
]

def _slug(url: str) -> str:
    s = url.split("https://about.gitlab.com/handbook/")[-1].strip("/")
    return (s or "index").replace("/", "-")

def have_any_markdown() -> bool:
    return any(PROC_DIR.rglob("*.md"))

def bootstrap_gitlab(progress: Optional[Prog] = None) -> Dict:
    """Download curated GitLab pages → HTML, convert to Markdown, write to data/processed."""
    if requests is None or md is None:
        return {"ok": False, "error": "requests/markdownify not available"}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    total = len(CURATED)
    ok = 0
    for i, url in enumerate(CURATED, 1):
        try:
            if progress:
                progress.update(label=f"Fetching {i}/{total}: {url}", value=min(0.15, i/total * 0.15))
            r = requests.get(url, headers={"User-Agent": "EKA-Streamlit/1.0"}, timeout=20)
            r.raise_for_status()
            html = r.text
            (RAW_DIR / f"{_slug(url)}.html").write_text(html, encoding="utf-8")
            (PROC_DIR / f"{_slug(url)}.md").write_text(md(html), encoding="utf-8")
            ok += 1
            time.sleep(0.05)
        except Exception:
            pass
    return {"ok": ok > 0, "downloaded": ok, "total": total}

# ---------------- UI chrome ----------------
st.set_page_config(page_title="EKA (Streamlit-only)", page_icon="✨", layout="wide")
st.markdown("""
<style>
.header-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.badge { border-radius:999px; padding:4px 10px; font-size:0.85rem; border:1px solid transparent; }
.badge-ok  { background:#E8FFF3; color:#05603A; border-color:#ABEFC6; }
.badge-warn{ background:#FFF7ED; color:#9A3412; border-color:#FED7AA; }
.kpi{padding:8px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#FCFEFF;display:inline-block;margin-right:8px;}
.small{font-size:0.92rem;color:#687076;}
.cv-pill { display:inline-block; margin:4px 6px 8px 0; padding:6px 10px; border-radius:999px;
           border:1px solid #e5e7eb; font-size:0.95rem; }
.cv-b1 { background:#FEE2E2; } .cv-b2 { background:#FEF9C3; }
.cv-b3 { background:#ECFEFF; } .cv-b4 { background:#F0FDF4; }
</style>
""", unsafe_allow_html=True)

# ---------------- Models & index ----------------
@st.cache_resource(show_spinner=False)
def get_models():
    emb = Embedder(EMB_MODEL, NORMALIZE)
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

def _scan_processed_files() -> List[pathlib.Path]:
    return sorted(PROC_DIR.glob("**/*.md"))

def _split_md(text: str) -> List[Dict]:
    return list(split_markdown(text, **CHUNK_CFG))

def rebuild_index(progress: Optional[Prog] = None) -> Dict:
    files = _scan_processed_files()
    if progress:
        progress.update(label=f"Scanning… {len(files)} files", value=0)
    records = []
    done = 0
    for i, fp in enumerate(files, 1):
        t = fp.read_text(encoding="utf-8")
        for ch in _split_md(t):
            records.append({"text": ch["text"], "source": str(fp).replace("\\","/")})
        done = i
        if progress:
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
            if progress:
                progress.update(label=f"Embedding… {e}/{len(texts)} chunks",
                                value=0.25 + 0.60*(e/max(1,len(texts))))
        X = np.vstack(vecs)
    else:
        import numpy as np
        X = np.zeros((0, 384), dtype="float32")

    idx = load_or_init_index()
    if progress:
        progress.update(label="Writing index…", value=0.9)
    idx.build(X, records)
    if progress:
        progress.update(label="Done", value=1.0)
    return {"num_files": len(files), "num_chunks": len(records)}

# ---------------- Retrieval & attribution ----------------
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

# ---------------- Header (after helpers; no side-effects yet) ----------------
st.markdown(
    "<div class='header-row'><div><h3>Enterprise Knowledge Agent (Streamlit-only)</h3>"
    f"<div class='small'>Runs fully inside Streamlit. {_openai_diag()}</div></div>"
    "<div><span class='badge badge-ok'>Online</span></div></div>",
    unsafe_allow_html=True
)

# ---------------- First-run bootstrap (now that functions exist) -------------
def corpus_stats() -> Dict:
    files = list(PROC_DIR.rglob("*.md"))
    n_files = len(files)
    n_bytes = sum((f.stat().st_size for f in files), 0)
    return {"files": n_files, "size_kb": int(n_bytes/1024)}

def first_run_bootstrap():
    if have_any_markdown():
        return
    st.warning("No corpus found — attempting to fetch the GitLab Handbook subset.", icon="⚠️")
    prog = Prog("Preparing…")
    did_any = False
    if AUTO_BOOTSTRAP:
        fetch_res = bootstrap_gitlab(progress=prog)
        if fetch_res.get("ok"):
            _ = rebuild_index(prog)
            st.success(f"Fetched {fetch_res.get('downloaded',0)} pages and built index.", icon="✅")
            did_any = True
    if not did_any:
        st.info("Bootstrap failed or disabled — using built-in seed corpus.", icon="ℹ️")
        write_seed_corpus()
        _ = rebuild_index(prog)
        st.success("Seed corpus indexed. You can now ask questions.", icon="✅")

# Self-heal guard: seed/rebuild/repair if needed
def ensure_ready_and_index(force_rebuild: bool = False) -> Dict:
    """Self-heal: if no processed markdown or index files are missing/empty, seed and rebuild."""
    idx_files_ok = pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists()

    if not have_any_markdown():
        # nothing to index → seed corpus then rebuild
        write_seed_corpus()
        prog = Prog("Indexing seed corpus…")
        stats = rebuild_index(prog)
        return {"seeded": True, "rebuilt": True, "stats": stats}

    # We have markdown; check index presence or forced rebuild
    if force_rebuild or not idx_files_ok:
        prog = Prog("Rebuilding index…")
        stats = rebuild_index(prog)
        return {"seeded": False, "rebuilt": True, "stats": stats}

    # As a final guard, attempt a tiny query; if it returns nothing, rebuild
    try:
        emb, _ = get_models()
        qv = emb.encode(["ping"])
        idx = load_or_init_index()
        has_any = idx.query_dense(qv, k=1)
        if not has_any:
            prog = Prog("Repairing empty index…")
            stats = rebuild_index(prog)
            return {"seeded": False, "rebuilt": True, "stats": stats}
    except Exception:
        prog = Prog("Repairing index…")
        stats = rebuild_index(prog)
        return {"seeded": False, "rebuilt": True, "stats": stats}

    return {"seeded": False, "rebuilt": False, "stats": corpus_stats()}

# Call AFTER all defs are in place
first_run_bootstrap()
_ = ensure_ready_and_index(force_rebuild=False)

# ---------------- Tabs & UI ----------------
tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

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
            # Self-heal before retrieve
            if not have_any_markdown():
                st.info("No corpus found — seeding and indexing now…")
                ensure_ready_and_index(force_rebuild=True)
            else:
                # If index files are missing or empty, rebuild
                idx_missing = not pathlib.Path(INDEX_PATH).exists() or not pathlib.Path(STORE_PATH).exists()
                if idx_missing:
                    st.info("Index not found — rebuilding now…")
                    ensure_ready_and_index(force_rebuild=True)

            recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)

            # If still empty, force one rebuild as a last resort
            if len(pairs) == 0:
                st.info("No sources returned — repairing index and retrying once…")
                ensure_ready_and_index(force_rebuild=True)
                recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)

            ans, llm_meta = generate_answer(recs, q, max_chars=max_chars)

            st.subheader("Answer")
            if not (ans or "").strip():
                st.warning("No text returned. Try ‘About → Use seed corpus’ then press ‘Rebuild index’.")
            else:
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
                title = f"Top source #{rk}\n{src}\n(score {s:.2f})"
                pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
            st.markdown("".join(pills), unsafe_allow_html=True)

            st.markdown("#### Sources")
            for c in fmt_citations(pairs, TOP_K):
                with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}"):
                    st.write(c['preview'])

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
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        for a in actions:
            if a.get("type") == "upsert_wiki_draft":
                slug = re.sub(r"[^a-z0-9\-]+", "-", a["title"].lower()).strip("-") or "page"
                (WIKI_DIR / f"{slug}.md").write_text(a["content"], encoding="utf-8")
                applied.append({"upserted": f"{slug}.md"})
        if applied:
            prog = Prog("Agent: updating index…")
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

with tab_upload:
    st.markdown("**Upload `.md` or `.txt` — added to the knowledge base.**")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & rebuild index"):
        if not f:
            st.warning("Please select a file first.")
        else:
            name = (ttl or f.name).strip() or "page"
            slug = re.sub(r"[^a-z0-9\-]+", "-", name.lower()).strip("-") or "page"
            WIKI_DIR.mkdir(parents=True, exist_ok=True)
            text = f.getvalue().decode("utf-8", errors="ignore")
            (WIKI_DIR / f"{slug}.md").write_text(text, encoding="utf-8")
            prog = Prog("Indexing upload…")
            stats = rebuild_index(prog)
            st.success(f"✅ Indexed {stats['num_chunks']} chunks from {stats['num_files']} files.")

with tab_about:
    st.markdown("""
### Controls & status
- **Rebuild index**: re-embeds & rewrites FAISS/TF-IDF from current `data/processed/**.md`  
- **Bootstrap**: try downloading the GitLab subset (if your runtime has outbound internet)  
- **Use seed corpus**: write a small, built-in corpus and index it (works fully offline)  
- **Corpus stats**: quick view of how much content is indexed
""")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Rebuild index"):
        prog = Prog("Rebuilding…")
        stats = rebuild_index(prog)
        st.success(f"Rebuilt: {stats['num_chunks']} chunks from {stats['num_files']} files.")
    if c2.button("Bootstrap (download)"):
        prog = Prog("Bootstrapping…")
        res = bootstrap_gitlab(progress=prog)
        if res.get("ok"):
            stats = rebuild_index(prog)
            st.success(f"Downloaded {res.get('downloaded',0)} pages; indexed {stats['num_chunks']} chunks.")
        else:
            st.error("Bootstrap failed (likely no internet or missing libs). Try 'Use seed corpus'.")
    if c3.button("Use seed corpus"):
        write_seed_corpus()
        prog = Prog("Indexing seed corpus…")
        stats = rebuild_index(prog)
        st.success(f"Seed indexed: {stats['num_chunks']} chunks from {stats['num_files']} files.")
    if c4.button("Quick repair (seed + rebuild)"):
        res = ensure_ready_and_index(force_rebuild=True)
        if res.get("rebuilt") or res.get("seeded"):
            st.success("Quick repair complete.")
        else:
            st.info("Index already healthy.")
    stats_now = corpus_stats()
    st.info(f"Corpus stats: {stats_now['files']} files · ~{stats_now['size_kb']} KB")
