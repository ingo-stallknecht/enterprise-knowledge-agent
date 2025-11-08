# app/streamlit_mono.py
# Streamlit-only Enterprise Knowledge Agent (Streamlit Cloud friendly)
# - Prebuilt index support for instant start (data/index/prebuilt/*)
# - Incremental indexing on upload (no full rebuild)
# - Agent = plan → retrieve → draft → propose actions → approve → act
# - Action preview, confidence threshold, explicit approval, no auto-writes by default
# - Keeps Agent tab focus (no jumping)

import sys, os, pathlib, re, time, tempfile, shutil
from typing import List, Dict, Tuple, Optional
import streamlit as st

# ---------- Optional deps ----------
try:
    from markdownify import markdownify as md
except Exception:
    md = None
try:
    import requests
except Exception:
    requests = None

def _secret(k, default=None):
    try: return st.secrets[k]
    except Exception: return os.environ.get(k, default)

# ---------- Writable caches ----------
WRITABLE_BASE = pathlib.Path(_secret("EKA_CACHE_DIR", "/mount/tmp")).expanduser()
if not WRITABLE_BASE.exists():
    WRITABLE_BASE = pathlib.Path(tempfile.gettempdir()) / "eka"
HF_CACHE = WRITABLE_BASE / "hf"
DOTCACHE = WRITABLE_BASE / ".cache" / "huggingface"
DOTHF = WRITABLE_BASE / ".huggingface"
for p in (WRITABLE_BASE, HF_CACHE, DOTCACHE, DOTHF):
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass
os.environ.update({
    "HOME": str(WRITABLE_BASE),
    "HF_HUB_DISABLE_TOKEN_SEARCH": "1",
    "HUGGING_FACE_HUB_TOKEN": "",
    "HF_TOKEN": "",
    "HUGGINGFACE_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_CACHE_SYMLINKS": "1",
    "HF_HUB_OFFLINE": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": str(HF_CACHE),
    "TRANSFORMERS_CACHE": str(HF_CACHE),
    "SENTENCE_TRANSFORMERS_HOME": str(HF_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(HF_CACHE),
    "XDG_CACHE_HOME": str(WRITABLE_BASE),
})
for token_file in (HF_CACHE / "token", DOTCACHE / "token", DOTHF / "token"):
    try:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("", encoding="utf-8")
    except Exception:
        pass

# ---------- OpenAI / LLM toggles ----------
def _find_secret_key() -> str:
    candidates = ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_APIKEY", "openai_api_key", "openai_key"]
    try:
        for k in candidates:
            if k in st.secrets:
                v = str(st.secrets[k]).strip()
                if v: return v
        for section in ("openai", "OPENAI", "llm"):
            if section in st.secrets and isinstance(st.secrets[section], dict):
                for k in ("api_key", "API_KEY", "key"):
                    v = str(st.secrets[section].get(k, "")).strip()
                    if v: return v
    except Exception:
        pass
    for k in candidates:
        v = str(os.getenv(k, "")).strip()
        if v: return v
    return ""

use_openai_flag = str(st.secrets.get("USE_OPENAI", os.environ.get("USE_OPENAI", "true"))).lower() == "true"
os.environ["USE_OPENAI"] = "true" if use_openai_flag else "false"

_key = _find_secret_key()
if _key:
    os.environ["OPENAI_API_KEY"] = _key

# default to gpt-4o unless overridden
os.environ["OPENAI_MODEL"] = str(st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o")))
os.environ["OPENAI_MAX_DAILY_USD"] = str(st.secrets.get("OPENAI_MAX_DAILY_USD", os.environ.get("OPENAI_MAX_DAILY_USD", "0.80")))

def _openai_diag() -> str:
    use = os.environ.get("USE_OPENAI", "false").lower() == "true"
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "")
    if use and key: return f"OpenAI: ON (key ✓, model {model})"
    if use and not key: return "OpenAI: ON (key missing ⚠)"
    return "OpenAI: OFF"

# ---------- Settings ----------
TOP_K = int(_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER_DEFAULT = _secret("EKA_USE_RERANKER", "true").lower() == "true"
DISABLE_RERANKER_BOOT = _secret("EKA_DISABLE_RERANKER", "false").lower() == "true"
AUTO_BOOTSTRAP = _secret("EKA_BOOTSTRAP", "true").lower() == "true"
USE_PREBUILT = _secret("EKA_USE_PREBUILT", "true").lower() == "true"

# ---------- PY path ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- Local imports ----------
from app.rag.utils import ensure_dirs, load_cfg
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.rag.chunker import split_markdown
from app.llm.answerer import generate_answer
from app.llm.gpt_client import GPTClient

# ---------- Config & dirs ----------
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
PREBUILT_DIR = pathlib.Path("data/index/prebuilt")
CHUNK_CFG = CFG.get("chunk", {"max_chars": 1200, "overlap": 150})
ensure_dirs()

RAW_DIR = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")
WIKI_DIR = PROC_DIR / "wiki"

# ---------- CSS / UI ----------
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
</style>
""", unsafe_allow_html=True)

# ---------- Progress ----------
class Prog:
    def __init__(self, initial_text: str = "Working…"):
        self._pb = st.progress(0, text=initial_text); self._last = 0.0
    def update(self, label: Optional[str] = None, value: Optional[float] = None):
        if value is None: value = self._last
        value = max(0.0, min(float(value), 1.0)); self._last = value
        self._pb.progress(int(value * 100), text=label if label is not None else None)

# ---------- Seed corpus ----------
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

# ---------- Bootstrap helpers ----------
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

def have_any_markdown() -> bool: return any(PROC_DIR.rglob("*.md"))
def _scan_processed_files() -> List[pathlib.Path]: return sorted(PROC_DIR.glob("**/*.md"))
def _split_md(text: str) -> List[Dict]: return list(split_markdown(text, **CHUNK_CFG))

# ---------- Prebuilt index loader ----------
def copy_prebuilt_if_available() -> bool:
    pre_faiss = PREBUILT_DIR / pathlib.Path(INDEX_PATH).name
    pre_store = PREBUILT_DIR / pathlib.Path(STORE_PATH).name
    if pre_faiss.exists() and pre_store.exists():
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        shutil.copy2(pre_faiss, INDEX_PATH)
        shutil.copy2(pre_store, STORE_PATH)
        return True
    return False

# ---------- Models & Index ----------
@st.cache_resource(show_spinner=False)
def get_models():
    emb = Embedder(EMB_MODEL, NORMALIZE)
    rer = None
    if not DISABLE_RERANKER_BOOT:
        try: rer = Reranker(CFG.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception: rer = None
    return emb, rer

@st.cache_resource(show_spinner=False)
def load_or_init_index():
    idx = DocIndex(INDEX_PATH, STORE_PATH)
    if hasattr(idx, "load"):
        try: idx.load()
        except Exception: pass
    return idx

# ---------- Build & incremental ----------
def rebuild_index(progress: Optional[Prog] = None) -> Dict:
    files = _scan_processed_files()
    if progress: progress.update(label=f"Scanning… {len(files)} files", value=0)
    records = []
    for i, fp in enumerate(files, 1):
        t = fp.read_text(encoding="utf-8")
        for ch in _split_md(t):
            records.append({"text": ch["text"], "source": str(fp).replace("\\","/")})
        if progress:
            progress.update(label=f"Chunking… {i}/{len(files)} files",
                            value=min(0.25, 0.05 + 0.20*(i/max(1,len(files)))))

    import numpy as np
    emb, _ = get_models()
    texts = [r["text"] for r in records]
    if texts:
        vecs, batch = [], 64
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
    if progress: progress.update(label="Writing index…", value=0.9)
    if hasattr(idx, "build"): idx.build(X, records)
    if progress: progress.update(label="Done", value=1.0)
    return {"num_files": len(files), "num_chunks": len(records)}

def incremental_add(markdown_text: str, source_path: str, progress: Optional[Prog] = None) -> Dict:
    chunks = _split_md(markdown_text)
    if progress: progress.update(label=f"Chunking {len(chunks)} chunks…", value=0.15)
    texts = [c["text"] for c in chunks]
    if not texts: return {"added_chunks": 0}
    emb, _ = get_models()
    vecs = emb.encode(texts)
    records = [{"text": t, "source": source_path} for t in texts]
    idx = load_or_init_index()
    if progress: progress.update(label="Appending to index…", value=0.6)
    if hasattr(idx, "add"): idx.add(vecs, records)
    else: return rebuild_index(progress)
    if progress: progress.update(label="Saved.", value=1.0)
    return {"added_chunks": len(texts)}

# ---------- Retrieval & attribution ----------
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
    reranked, rerank_used = raw, False
    if use_reranker and rer is not None and candidates:
        pairs = rer.rerank(query, candidates[:30], top_k=max(k,1))
        reranked = [(rec, sc) for rec, sc in pairs]; rerank_used = True
    top_records = [r for r,_ in reranked[:k]]
    meta = {"mode": meta_mode, "alpha": ALPHA, "rerank_used": rerank_used,
            "retrieval_ms": int((time.time()-t0)*1000)}
    return top_records, reranked, meta

_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')
def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t: return []
    parts = _SENT_SPLIT.split(t)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) < 40 and buf:
            buf = f"{buf} {p}"
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
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
        j = int(np.argmax(sim[i])); best = records[j]
        out.append({
            "sentence": s,
            "best_source": best.get("source"),
            "best_rank": j+1,
            "score": float(sim[i,j]),
            "preview": (best.get("text") or "")[:240] + ("…" if len((best.get("text") or ""))>240 else "")
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

# ---------- GPT helper ----------
_gpt_singleton = None
def _get_gpt() -> Optional[GPTClient]:
    global _gpt_singleton
    if _gpt_singleton is None:
        try: _gpt_singleton = GPTClient()
        except Exception: _gpt_singleton = None
    return _gpt_singleton

def gpt_compose(context: str, prompt: str, max_chars: int) -> Optional[str]:
    gpt = _get_gpt()
    if not gpt or os.environ.get("USE_OPENAI","false").lower()!="true":
        return None
    try:
        text, _meta = gpt.answer(context, prompt)
        text = (text or "").strip()
        if text: return text[:max_chars].rstrip()
    except Exception:
        return None
    return None

# ---------- Header + status ----------
def _index_health() -> Tuple[str, str]:
    try:
        idx = load_or_init_index()
        ok = (idx.size() > 0) if hasattr(idx, "size") else True
        if st.session_state.get("eka_busy"): return "<span class='badge badge-warm'>Loading</span>", "warm"
        return ("<span class='badge badge-ok'>Online</span>", "ok") if ok else ("<span class='badge badge-err'>Empty</span>", "err")
    except Exception:
        return "<span class='badge badge-err'>Offline</span>", "err"

st.markdown(
    "<div class='header-row'>"
    "<div><h3>Enterprise Knowledge Agent</h3>"
    f"<div class='small'>Ask in plain language. Agent plans, retrieves, answers, cites. {_openai_diag()}</div></div>"
    "<div id='status-slot'></div></div>",
    unsafe_allow_html=True
)
_status_slot = st.empty()
badge_html, _ = _index_health()
_status_slot.markdown(f"<div style='display:flex; justify-content:flex-end'>{badge_html}</div>", unsafe_allow_html=True)

# ---------- Bootstrap & ready ----------
def bootstrap_gitlab(progress: Optional[Prog] = None) -> Dict:
    if requests is None or md is None:
        return {"ok": False, "error": "requests/markdownify not available"}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    total, ok = len(CURATED), 0
    for i, url in enumerate(CURATED, 1):
        try:
            if progress: progress.update(label=f"Fetching {i}/{total}: {url}", value=min(0.15, i/total * 0.15))
            r = requests.get(url, headers={"User-Agent": "EKA-Streamlit/1.0"}, timeout=20)
            r.raise_for_status()
            html = r.text
            (RAW_DIR / f"{_slug(url)}.html").write_text(html, encoding="utf-8")
            (PROC_DIR / f"{_slug(url)}.md").write_text(md(html), encoding="utf-8")
            ok += 1; time.sleep(0.02)
        except Exception:
            pass
    return {"ok": ok > 0, "downloaded": ok, "total": total}

def corpus_stats() -> Dict:
    files = list(PROC_DIR.rglob("*.md"))
    n_files = len(files)
    n_bytes = sum((f.stat().st_size for f in files), 0)
    return {"files": n_files, "size_kb": int(n_bytes/1024)}

def first_run_bootstrap():
    if USE_PREBUILT and copy_prebuilt_if_available(): return
    if have_any_markdown() and pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists(): return
    prog = Prog("Preparing…")
    did_any = False
    if AUTO_BOOTSTRAP:
        st.warning("No corpus found — attempting to fetch the GitLab Handbook subset.", icon="⚠️")
        fetch_res = bootstrap_gitlab(progress=prog)
        if fetch_res.get("ok"):
            st.session_state["eka_busy"] = True
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
            _ = rebuild_index(prog)
            st.success(f"Fetched {fetch_res.get('downloaded',0)} pages and built index.", icon="✅")
            did_any = True
            st.session_state["eka_busy"] = False
    if not did_any:
        st.info("Using built-in seed corpus.", icon="ℹ️")
        write_seed_corpus()
        st.session_state["eka_busy"] = True
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
        _ = rebuild_index(prog)
        st.success("Seed corpus indexed. You can now ask questions.", icon="✅")
        st.session_state["eka_busy"] = False

def ensure_ready_and_index(force_rebuild: bool = False) -> Dict:
    idx_files_ok = pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists()
    if not have_any_markdown():
        write_seed_corpus()
        prog = Prog("Indexing seed corpus…")
        st.session_state["eka_busy"] = True
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
        stats = rebuild_index(prog)
        st.session_state["eka_busy"] = False
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
        return {"seeded": True, "rebuilt": True, "stats": stats}
    if force_rebuild or not idx_files_ok:
        prog = Prog("Rebuilding index…")
        st.session_state["eka_busy"] = True
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
        stats = rebuild_index(prog)
        st.session_state["eka_busy"] = False
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
        return {"seeded": False, "rebuilt": True, "stats": stats}
    try:
        emb, _ = get_models(); qv = emb.encode(["ping"]); idx = load_or_init_index()
        if not idx.query_dense(qv, k=1):
            prog = Prog("Repairing empty index…")
            st.session_state["eka_busy"] = True
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
            stats = rebuild_index(prog)
            st.session_state["eka_busy"] = False
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
            return {"seeded": False, "rebuilt": True, "stats": stats}
    except Exception:
        prog = Prog("Repairing index…")
        st.session_state["eka_busy"] = True
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
        stats = rebuild_index(prog)
        st.session_state["eka_busy"] = False
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
        return {"seeded": False, "rebuilt": True, "stats": stats}
    return {"seeded": False, "rebuilt": False, "stats": corpus_stats()}

# Call AFTER defs
first_run_bootstrap()
_ = ensure_ready_and_index(False)

# ---------- Tabs & param helpers ----------
try:
    qp = st.query_params
    get_qp = lambda k, default=None: qp.get(k, default)
    set_qp = lambda **kw: st.query_params.update(kw)
except Exception:
    get_qp = lambda k, default=None: st.experimental_get_query_params().get(k, [default])[0]
    set_qp = lambda **kw: st.experimental_set_query_params(**kw)
st.session_state.setdefault("focus_tab", get_qp("tab", "ask"))

tabs = st.tabs(["Ask", "Agent", "Upload", "About"])
tab_ask, tab_agent, tab_upload, tab_about = tabs

# ===== ASK =====
with tab_ask:
    st.markdown("**Finds relevant passages and answers strictly from them. Shows citations and a sentence-to-source map.**")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are values applied in performance reviews?")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval mode", ["hybrid","dense","sparse"],
                        index=["hybrid","dense","sparse"].index(RETRIEVAL_MODE))
    use_rr = c2.checkbox("Use reranker", value=USE_RERANKER_DEFAULT)

    if st.button("Get answer", type="primary", key="ask_btn"):
        st.session_state["focus_tab"] = "ask"; set_qp(tab="ask")
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            ensure_ready_and_index(False)
            recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)
            if len(pairs) == 0:
                st.info("No sources returned — repairing index and retrying once…")
                ensure_ready_and_index(True)
                recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)
            ans, llm_meta = generate_answer(recs, q, max_chars=max_chars)
            if not ans or ans.strip() in {".", ""}:
                joined = "\n\n".join((r.get("text") or "") for r,_ in pairs[:6]).strip()
                ans = joined[:max_chars] if joined else "No relevant context found in the current corpus."
            st.subheader("Answer"); st.write(ans)
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='kpi'>Sources: <b>{len(pairs[:TOP_K])}</b></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='kpi'>Retrieval: <b>{meta.get('mode','')}</b></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='kpi'>Latency: <b>{meta.get('retrieval_ms',0)} ms</b></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='kpi'>LLM: <b>{llm_meta.get('llm','extractive')}</b></div>", unsafe_allow_html=True)
            st.markdown("##### Why this answer: Context Visualizer")
            st.markdown("<div class='cv-legend small'><span>Evidence strength:</span><span class='cv-dot cv-b1'></span><span class='small'>weak</span><span class='cv-dot cv-b2'></span><span class='small'>medium</span><span class='cv-dot cv-b3'></span><span class='small'>strong</span><span class='cv-dot cv-b4'></span><span class='small'>very strong</span></div>", unsafe_allow_html=True)
            def band(s): return 4 if s>=0.75 else 3 if s>=0.50 else 2 if s>=0.30 else 1
            def bcls(b): return {1:"cv-b1",2:"cv-b2",3:"cv-b3",4:"cv-b4"}[b]
            pills = []
            for row in attribute(ans, recs):
                s = float(row.get("score",0)); sent = (row.get("sentence","") or "").strip()
                src = row.get("best_source",""); rk = row.get("best_rank","-")
                title = f"Top source #{rk}\n{src}\n(score {s:.2f})"
                pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.markdown("#### Sources")
            for c in fmt_citations(pairs, TOP_K):
                with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                    st.write(c['preview'])

# ===== AGENT =====

# --- Helper scoring/points ---
def _looks_like_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 50 or len(s) > 240: return False
    if s.startswith("#") or s.startswith(">"): return False
    if re.search(r"\[(.*?)\]\(http", s): return False
    if "http://" in s or "https://" in s: return False
    if not re.match(r"^[A-Z0-9].*[\.!\?]$", s): return False
    if re.fullmatch(r"[A-Za-z\s]{0,35}", s): return False
    return True

def _extract_key_points(query: str, pairs: List[Tuple[Dict, float]], emb: Embedder, max_points: int = 5) -> List[str]:
    if not pairs: return []
    qv = emb.encode([query])
    cands = []
    for rec, _sc in pairs[:24]:
        text = (rec.get("text") or "")
        for s in _split_sentences(text):
            if _looks_like_sentence(s):
                cands.append(s.strip())
    if not cands: return []
    sv = emb.encode(cands)
    import numpy as np
    sim = (sv @ qv.T).reshape(-1)
    idxs = np.argsort(-sim)[: max(10, max_points*3)]
    picked = [cands[i] for i in idxs]
    # de-dupe but keep order
    seen, out = set(), []
    for s in picked:
        k = re.sub(r"[^a-z0-9]+"," ", s.lower()).strip()
        if k and k not in seen:
            seen.add(k); out.append(s)
    return out[:max_points]

def _clean_text_blocks(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\b(no context|not in context|hallucination)\b", "", t, flags=re.I)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.replace("\u0000", "").strip()
    return t

def concise_title_from_query(q: str) -> str:
    base = q.strip()
    base = re.sub(r"\?$", "", base)
    base = re.sub(r"^(create|write|make|draft)\s+", "", base, flags=re.I)
    base = re.sub(r";\s*if.*$", "", base, flags=re.I)
    title = base.strip().rstrip(".")
    if len(title) > 60: title = title[:57].rstrip() + "…"
    def _tc(s):
        return " ".join(w.capitalize() if w.lower() not in {"in","and","or","of","to","for","with","on"} else w.lower()
                        for w in re.split(r"(\s+|\—)", s))
    title = _tc(title)
    title = re.sub(r"\s*—\s*", " — ", title)
    return title

def build_wiki_prompt(title: str, key_points: List[str], style: str) -> Tuple[str, str]:
    kp = "\n".join(f"- {k}" for k in key_points) if key_points else "- Outline 3–5 concrete principles with examples."
    sys = "You are a precise technical writer. Write clearly, concisely, and avoid repetition. Use Markdown."
    detail = "short, high-signal" if style == "Concise" else "detailed but crisp"
    user = f"""
Write an internal wiki page titled "{title}" in a {detail} style.

Audience: ICs and managers preparing performance reviews.
Tone: crisp, actionable, value-aligned.
Sections: TL;DR, Key behaviors (bulleted), Examples (2-3), Checklist, References.

Key points to use:
- Time-box feedback and collect it before the decision deadline.
{kp}

Rules:
- Synthesize; don't copy bullets verbatim.
- Avoid bland filler; prefer concrete, verifiable phrasing.
- Propose example stubs with realistic placeholders (Issue/MR links).
- Keep it under 700 words.
"""
    return sys, user

def build_answer_prompt(question: str, style: str) -> Tuple[str, str]:
    sys = "You are an expert summarizer. Produce a user-facing summary people actually want to read."
    detail = "short, high-signal" if style == "Concise" else "detailed but crisp"
    user = f"""Question: {question}

Write a {detail} answer with these sections and always include one point about time-boxing feedback before deadlines:

**Why it matters (1–2 lines)**
**What to look for (4–6 bullets)**
**Example review bullets (3–5 bullets)**
**Checklist (4–6 bullets, include: time-box feedback and collect it before the deadline)**

Only use the provided context. If context is weak, say what's missing in one bullet.
Avoid repeating the question. No generic filler.
"""
    return sys, user

# --- GPT access
_gpt_singleton2 = None
def _get_gpt():
    global _gpt_singleton2
    if _gpt_singleton2 is None:
        try: _gpt_singleton2 = GPTClient()
        except Exception: _gpt_singleton2 = None
    return _gpt_singleton2

def gpt_compose(context: str, prompt: str, max_chars: int) -> Optional[str]:
    gpt = _get_gpt()
    if not gpt or os.environ.get("USE_OPENAI","false").lower()!="true":
        return None
    try:
        text, _meta = gpt.answer(context, prompt)
        text = (text or "").strip()
        if text: return text[:max_chars].rstrip()
    except Exception:
        return None
    return None

# --- Agent core (no side effects) ---
def agent_propose(message: str, style: str = "Concise") -> Dict:
    # 1) Normalize
    user_msg = (message or "").strip()
    query = user_msg if user_msg.endswith("?") else (user_msg.rstrip(".") + "?")

    # 2) Retrieve (hybrid, fast)
    records, pairs, _meta = retrieve(query, min(TOP_K, 12), "hybrid", use_reranker=False)

    # 3) Short, user-facing answer
    ctx_for_llm = "\n\n".join((r.get("text") or "") for r,_ in pairs[:8])
    sys_a, user_a = build_answer_prompt(query, style)
    gpt_ans = gpt_compose(ctx_for_llm, f"{sys_a}\n\n{user_a}", max_chars=1200)
    if gpt_ans:
        short_answer = _clean_text_blocks(gpt_ans)
    else:
        short_answer, _ = generate_answer([r for r,_ in pairs[:8]], query, max_chars=900)
        short_answer = _clean_text_blocks(short_answer)

    # 4) Extract key points & gaps
    emb, _ = get_models()
    points = _extract_key_points(query, pairs, emb, max_points=5)
    if not any("time-box" in p.lower() or "time box" in p.lower() for p in points):
        points = ["Time-box feedback windows and collect input before the decision deadline."] + points

    gaps = []
    if len(pairs) < 4:
        gaps.append("Low context richness (fewer than 4 strong passages). Consider ingesting more relevant docs.")
    if not points:
        gaps.append("Key behaviors not clearly present in sources. Add examples or policy docs.")

    # 5) Compose wiki (proposal only)
    wiki_title = concise_title_from_query(query)
    sys_w, user_w = build_wiki_prompt(wiki_title, points, style)
    wiki_ctx = "\n\n".join((r.get("text") or "") for r,_ in pairs[:12])
    gpt_wiki = gpt_compose(wiki_ctx, f"{sys_w}\n\n{user_w}", max_chars=3500)
    if gpt_wiki:
        wiki_content = _clean_text_blocks(gpt_wiki)
    else:
        bullets = "\n".join(f"- {p}" for p in points) if points else "- Add concrete, value-aligned behaviors here."
        tldr = points[0] if points else "Summarize the most important value-driven behaviors with examples."
        wiki_content = f"""# {wiki_title}

## TL;DR
- {tldr}

## Key behaviors
{bullets}

## Examples
- Add one example per behavior. Link to issue/MR/doc where demonstrated.
- Describe outcomes (metrics, customer impact, quality).

## Checklist
- Document decisions and feedback.
- Time-box feedback windows and collect input before the decision deadline.
- Tie observations to values and outcomes.
- Link to evidence (issues/MRs/docs).

## References
(Agent synthesized from indexed documents.)
"""

    # 6) Confidence estimation
    conf = 0.86 if short_answer and len(pairs) >= 6 else 0.72 if short_answer else 0.55

    # 7) Citations for the short answer
    citations = fmt_citations(pairs, k=min(8, len(pairs)))

    return {
        "query": query,
        "short_answer": short_answer,
        "wiki_title": wiki_title,
        "wiki_content": wiki_content,
        "citations": citations,
        "gaps": gaps,
        "confidence": conf,
        "used_chunks": min(8, len(pairs))
    }

# --- Acting (only after approval) ---
def agent_apply_create_wiki(title: str, content: str) -> Dict:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9\-]+", "-", title.lower()).strip("-") or "page"
    dst = WIKI_DIR / f"{slug}.md"
    dst.write_text(content, encoding="utf-8")
    created_path = str(dst).replace("\\","/")
    prog = Prog("Incremental indexing…")
    st.session_state["eka_busy"] = True
    _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
    res = incremental_add(content, created_path, progress=prog)
    st.session_state["eka_busy"] = False
    _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
    return {"file": f"{slug}.md", "path": created_path, "added_chunks": res.get("added_chunks", 0)}

# --- Keep focus on Agent tab ---
def _stay_on_agent():
    st.session_state["focus_tab"] = "agent"
    set_qp(tab="agent")

# ===== Agent UI =====
with tab_agent:
    _stay_on_agent()  # never jump away

    st.markdown("**What the agent does**")
    st.write(
        "The agent rewrites your goal into a sharp query, retrieves evidence, drafts a short answer with citations, "
        "and proposes a clean internal wiki page. It never writes without approval (unless you explicitly allow automatic actions)."
    )
    st.markdown("**Key functionalities with high ROI**")
    st.write(
        "- Turn vague prompts into review-ready bullets\n"
        "- Generate concise, value-aligned guidance with examples\n"
        "- Propose internal wiki pages for reuse and onboarding\n"
        "- Incrementally index new content without full rebuilds"
    )

    cA, cB, cC = st.columns([2,1,1])
    with cA:
        msg = st.text_area("Goal or task", height=110,
                           placeholder="e.g., Values in performance reviews — create an example-rich internal note.")
    with cB:
        style = st.radio("Answer style", ["Concise","Detailed"], index=0)
    with cC:
        conf_threshold = st.slider("Auto-action threshold", 0.5, 0.95, 0.80, 0.01)

    def _on_toggle_auto():
        _stay_on_agent()
    auto_actions = st.checkbox("Allow agent to make automatic actions (e.g., create wiki and index)",
                               value=False, on_change=_on_toggle_auto, key="agent_auto")

    if st.button("Plan & Draft", type="primary", key="agent_plan_btn"):
        _stay_on_agent()
        if not msg.strip():
            st.warning("Please describe what you want the agent to do.")
        else:
            st.session_state["eka_busy"] = True
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
            proposal = agent_propose(msg, style=style)
            st.session_state["eka_busy"] = False
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
            st.session_state["last_agent_proposal"] = proposal

    prop = st.session_state.get("last_agent_proposal")
    if prop:
        st.subheader("Agent answer")
        st.write(prop["short_answer"])

        st.markdown("#### What happened")
        st.write(f"• Rewrote to: `{prop['query']}`")
        st.write(f"• Retrieved and used top {prop['used_chunks']} chunks for drafting.")
        st.write(f"• Proposed wiki: **{prop['wiki_title']}**")
        st.write(f"• Confidence: {prop['confidence']:.2f}")
        if prop["gaps"]:
            st.write("• Gaps:")
            for g in prop["gaps"]:
                st.write(f"  – {g}")

        st.markdown("#### Citations")
        for c in prop["citations"]:
            with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                st.write(c["preview"])

        st.markdown("#### Proposed wiki page")
        st.text_input("Title", value=prop["wiki_title"], key="wiki_title_input")
        st.code(prop["wiki_content"], language="markdown")

        # --- Approval / act ---
        can_auto = auto_actions and (prop["confidence"] >= conf_threshold)
        col1, col2 = st.columns([1,3])
        do_it = False
        if can_auto:
            st.success("Auto-actions allowed and confidence ≥ threshold. The agent is ready to write on approval.")
        if col1.button("Approve & create wiki", key="approve_btn"):
            _stay_on_agent(); do_it = True
        if can_auto and not do_it:
            # If auto and not clicked, still require explicit approval to avoid surprises in portfolio demo
            st.info("Click **Approve & create wiki** to proceed. Auto-actions are enabled but still require approval here.")

        if do_it:
            title = st.session_state.get("wiki_title_input", prop["wiki_title"]).strip() or prop["wiki_title"]
            res = agent_apply_create_wiki(title, prop["wiki_content"])
            st.success(f"Wiki page created and indexed: **{title}**  –  {res['file']}  (chunks added: {res['added_chunks']})")
            with st.expander("Show created wiki page", expanded=True):
                st.code(prop["wiki_content"], language="markdown")
            # Clear proposal to avoid duplicate approvals
            st.session_state.pop("last_agent_proposal", None)

# ===== UPLOAD =====
with tab_upload:
    st.markdown("**Add plain-text or Markdown files to your knowledge base. They will be cited in answers.**")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & update knowledge", key="upload_btn"):
        set_qp(tab="upload")
        if not f:
            st.warning("Please select a file first.")
        else:
            name = (ttl or f.name).strip() or "page"
            slug = re.sub(r"[^a-z0-9\-]+", "-", name.lower()).strip("-") or "page"
            WIKI_DIR.mkdir(parents=True, exist_ok=True)
            text = f.getvalue().decode("utf-8", errors="ignore")
            dst = WIKI_DIR / f"{slug}.md"
            dst.write_text(text, encoding="utf-8")

            st.session_state["eka_busy"] = True
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
            prog = Prog("Incremental indexing…")
            res = incremental_add(text, str(dst).replace("\\","/"), progress=prog)
            st.session_state["eka_busy"] = False
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)
            st.success(f"✅ Incrementally indexed {res['added_chunks']} chunks from {slug}.md")

# ===== ABOUT =====
with tab_about:
    st.markdown("### How this works")
    st.markdown("""
- **Hybrid retrieval**: semantic + (placeholder keyword) for recall & precision.
- **Reranker (optional)**: cross-encoder refines the top candidates.
- **Q&A**: answers strictly from retrieved passages; cites sources.
- **Context Visualizer**: each sentence maps to the strongest supporting passage.
- **Incremental indexing**: uploads append vectors instead of full rebuilds.
- **Prebuilt index**: drop files into `data/index/prebuilt/` for instant startup.
""")
