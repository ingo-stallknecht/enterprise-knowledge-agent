# app/streamlit_mono.py
# Streamlit-only Enterprise Knowledge Agent (Streamlit Cloud friendly)
# - Navigation with a single radio control (prevents switching back to "Ask" after Agent actions)
# - GPT authoring step for the Agent (clean, example-rich content) with robust fallback
# - Incremental indexing on upload (no full rebuild), prebuilt index support
# - Polished UI (badges, KPIs, pills) and safer text cleaning

import sys, os, pathlib, re, time, tempfile, shutil
from typing import List, Dict, Tuple, Optional
import streamlit as st

# Optional HTML→Markdown
try:
    from markdownify import markdownify as md
except Exception:
    md = None

# Optional fetch
try:
    import requests
except Exception:
    requests = None

# ---------- Secrets / ENV helpers ----------
def _secret(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)

# ---------- Writable caches (Streamlit Cloud safe) ----------
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

os.environ["OPENAI_MODEL"] = str(st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
os.environ["OPENAI_MAX_DAILY_USD"] = str(st.secrets.get("OPENAI_MAX_DAILY_USD", os.environ.get("OPENAI_MAX_DAILY_USD", "0.50")))

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

# Local imports
from app.rag.utils import ensure_dirs, load_cfg
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.rag.chunker import split_markdown
from app.llm.answerer import generate_answer
# direct GPT access for agent authoring
try:
    from app.llm.gpt_client import GPTClient
except Exception:
    GPTClient = None  # fallback handled below

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
ensure_dirs()  # creates data/raw, data/processed/wiki, data/index, data/billing

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

# ---------- Progress wrapper ----------
class Prog:
    def __init__(self, initial_text: str = "Working…"):
        self._pb = st.progress(0, text=initial_text); self._last = 0.0
    def update(self, label: Optional[str] = None, value: Optional[float] = None):
        if value is None: value = self._last
        value = max(0.0, min(float(value), 1.0)); self._last = value
        self._pb.progress(int(value * 100), text=label if label is not None else None)

# ---------- Seed corpus (offline) ----------
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

def have_any_markdown() -> bool:
    return any(PROC_DIR.rglob("*.md"))

def _scan_processed_files() -> List[pathlib.Path]:
    return sorted(PROC_DIR.glob("**/*.md"))

def _split_md(text: str) -> List[Dict]:
    return list(split_markdown(text, **CHUNK_CFG))

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
        try:
            rer = Reranker(CFG.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception:
            rer = None
    return emb, rer

@st.cache_resource(show_spinner=False)
def load_or_init_index():
    idx = DocIndex(INDEX_PATH, STORE_PATH)
    # optional load() / size() methods if available
    try:
        if hasattr(idx, "load"): idx.load()
    except Exception:
        pass
    return idx

# ---------- Build & incremental index ----------
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
    if progress: progress.update(label="Writing index…", value=0.9)
    if hasattr(idx, "build"):
        idx.build(X, records)
    if progress: progress.update(label="Done", value=1.0)
    return {"num_files": len(files), "num_chunks": len(records)}

def incremental_add(markdown_text: str, source_path: str, progress: Optional[Prog] = None) -> Dict:
    chunks = _split_md(markdown_text)
    if progress: progress.update(label=f"Chunking {len(chunks)} chunks…", value=0.15)
    texts = [c["text"] for c in chunks]
    if not texts:
        return {"added_chunks": 0}
    emb, _ = get_models()
    vecs = emb.encode(texts)
    records = [{"text": t, "source": source_path} for t in texts]
    idx = load_or_init_index()
    if progress: progress.update(label="Appending to index…", value=0.6)
    if hasattr(idx, "add"):
        idx.add(vecs, records)
    else:
        return rebuild_index(progress)
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
        pairs = rer.rerank(query, candidates[:30], top_k=max(k,1))  # speed cap
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
    # 1) prebuilt → instant
    if USE_PREBUILT and copy_prebuilt_if_available():
        return
    # 2) live good?
    if have_any_markdown() and pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists():
        return
    # 3) bootstrap or seed
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
    # probe dense
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

# run bootstrap
first_run_bootstrap()
_ = ensure_ready_and_index(False)

# ---------- Navigation (prevents jumping back to first tab) ----------
if "nav" not in st.session_state:
    st.session_state["nav"] = "Ask"
nav = st.radio(" ", ["Ask", "Agent", "Upload", "About"], horizontal=True, label_visibility="collapsed", index=["Ask","Agent","Upload","About"].index(st.session_state["nav"]))
st.session_state["nav"] = nav

# ===== ASK =====
if nav == "Ask":
    st.markdown("**Finds relevant passages and answers strictly from them. Shows citations and a sentence-to-source map.**")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are values applied in performance reviews?")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval mode", ["hybrid","dense","sparse"],
                        index=["hybrid","dense","sparse"].index(RETRIEVAL_MODE))
    use_rr = c2.checkbox("Use reranker", value=USE_RERANKER_DEFAULT)

    if st.button("Get answer", type="primary"):
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

            # Legend + pills
            st.markdown("##### Why this answer: Context Visualizer")
            st.markdown("<div class='cv-legend small'><span>Evidence strength:</span><span class='cv-dot cv-b1'></span><span class='small'>weak</span><span class='cv-dot cv-b2'></span><span class='small'>medium</span><span class='cv-dot cv-b3'></span><span class='small'>strong</span><span class='cv-dot cv-b4'></span><span class='small'>very strong</span></div>", unsafe_allow_html=True)
            def band(s): return 4 if s>=0.75 else 3 if s>=0.50 else 2 if s>=0.30 else 1
            def bcls(b): return {1:"cv-b1",2:"cv-b2",3:"cv-b3",4:"cv-b4"}[b]
            pills = []
            for row in attribute(ans, recs):
                s = float(row.get("score",0))
                sent = (row.get("sentence","") or "").strip()
                src = row.get("best_source",""); rk = row.get("best_rank","-")
                title = f"Top source #{rk}\n{src}\n(score {s:.2f})"
                pills.append(f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>")
            st.markdown("".join(pills), unsafe_allow_html=True)

            st.markdown("#### Sources")
            for c in fmt_citations(pairs, TOP_K):
                with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                    st.write(c['preview'])

# ===== AGENT =====
def _looks_like_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 50 or len(s) > 240: return False
    if s.startswith("#") or s.startswith(">"): return False
    if re.search(r"\[(.*?)\]\(http", s): return False  # markdown link
    if "http://" in s or "https://" in s: return False
    if not re.match(r"^[A-Z0-9].*[\.!\?]$", s): return False
    if re.fullmatch(r"[A-Za-z\\s]{0,35}", s): return False
    # simple stoplist for noisy handbook headers
    if re.search(r"\b(Key principles|Engineering Workflow|Write things down)\b", s, flags=re.I): return False
    return True

def _dedupe_keep_order(strings: List[str]) -> List[str]:
    seen, out = set(), []
    for s in strings:
        k = re.sub(r"[^a-z0-9]+"," ", s.lower()).strip()
        if k and k not in seen:
            seen.add(k); out.append(s)
    return out

def _extract_key_points(query: str, pairs: List[Tuple[Dict, float]], emb: Embedder, max_points: int = 6) -> List[str]:
    if not pairs: return []
    qv = emb.encode([query])
    cands = []
    for rec, _sc in pairs[:30]:
        text = (rec.get("text") or "")
        for s in _split_sentences(text):
            if _looks_like_sentence(s):
                cands.append(s.strip())
    if not cands: return []
    sv = emb.encode(cands)
    import numpy as np
    sim = (sv @ qv.T).reshape(-1)
    idxs = np.argsort(-sim)[: max(12, max_points*3)]
    best = [cands[i] for i in idxs]
    best = _dedupe_keep_order(best)
    return best[:max_points]

def _clean_text(t: str, max_len: int = 1200) -> str:
    if not t: return ""
    # remove obvious noise and nulls
    t = t.replace("\x00", " ").replace("NULL", " ")
    t = re.sub(r"\b(no context|not in context)\b", "", t, flags=re.I)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_len: t = t[:max_len].rstrip() + "…"
    return t

def _make_wiki_template(topic: str, key_points: List[str]) -> str:
    bullets = "\n".join(f"- {kp}" for kp in key_points[:6]) if key_points else "- Add concrete principles from your domain."
    tldr = key_points[0] if key_points else "Summarize the most important value-driven behaviors with examples."
    return f"""# {topic}

> Internal guide – drafted by the Knowledge Agent. Please review before publishing.

## TL;DR
- {tldr}

## Key principles & behaviors
{bullets}

## Examples (fill with concrete evidence)
- Add one example per behavior. Link to issue/MR/doc where the behavior is demonstrated.
- Describe outcomes (metrics, customer impact, quality).

## References
(These notes were derived from internal documents indexed by the agent.)
"""

def _gpt_author_note(chunks: List[Dict], goal: str, max_len: int = 900) -> str:
    """Use GPT directly to write a clean, example-rich note from retrieved chunks."""
    use = os.environ.get("USE_OPENAI", "false").lower() == "true"
    if not use or GPTClient is None:
        return ""
    try:
        client = GPTClient()
        context = "\n\n".join((c.get("text") or "") for c in chunks[:8])
        prompt = (
            "Write a concise, example-rich note strictly from the context.\n"
            "Format with short paragraphs and bullets. Avoid headers like 'Key principles'.\n"
            "Cite nothing explicitly; keep it internal-facing and practical.\n\n"
            f"Goal: {goal}\n\n"
            f"Context:\n{context}\n"
        )
        # We bypass generate_answer to give a more precise task prompt.
        ans, meta = client.answer(context=prompt, question="")
        return _clean_text(ans, max_len=max_len)
    except Exception:
        return ""

def _synthesize_from_points(original_message: str, points: List[str], max_len: int = 900) -> str:
    if not points:
        return "No strong evidence found in the current corpus. Try seeding or bootstrapping the handbook."
    cleaned_q = re.sub(r"\?$", "", (original_message or "").strip())
    intro = f"**Short answer to:** {cleaned_q}."
    body = "\n".join(f"- {p}" for p in points[:8])
    txt = f"{intro}\n\n**Key points from the corpus:**\n{body}"
    return _clean_text(txt, max_len=max_len)

def agent_run(message: str, auto_actions: bool = False) -> Dict:
    # 1) Rewrite query for retrieval
    user_msg = (message or "").strip()
    query = user_msg
    if len(query) < 12 or not query.endswith("?"):
        query = (query.rstrip(".") + "?")

    # 2) Retrieve (limit for speed)
    records, pairs, _meta = retrieve(query, min(TOP_K, 12), "hybrid", use_reranker=False)

    # 3) Draft via GPT author step; fallback to general generate_answer, then to points
    authored = _gpt_author_note(records, user_msg, max_len=900)
    if not authored:
        draft, _ = generate_answer(records[:8], query, max_chars=900)
        authored = _clean_text(draft, max_len=900)
    if not authored or authored in {".", ""}:
        emb, _ = get_models()
        points = _extract_key_points(query, pairs, emb, max_points=6)
        authored = _synthesize_from_points(user_msg, points, max_len=900)
    else:
        # also compute points for wiki even if GPT succeeded
        emb, _ = get_models()
        points = _extract_key_points(query, pairs, emb, max_points=6)

    # 4) Wiki draft (deterministic, clean)
    wiki_title = "Guide: " + re.sub(r"\?$", "", query).strip().capitalize()
    wiki_content = _make_wiki_template(wiki_title, points)

    applied = []
    created_path = None
    if auto_actions:
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9\-]+", "-", wiki_title.lower()).strip("-") or "page"
        dst = WIKI_DIR / f"{slug}.md"
        dst.write_text(wiki_content, encoding="utf-8")
        applied.append({"upserted": f"{slug}.md"})
        created_path = str(dst).replace("\\","/")

        # Incremental embed of the new draft
        prog = Prog("Updating index…")
        st.session_state["eka_busy"] = True
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
        _ = incremental_add(wiki_content, created_path, progress=prog)
        st.session_state["eka_busy"] = False
        _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)

    # Minimal, human-readable trace
    trace = [
        {"step": 1, "action": "rewrite_query", "output": query},
        {"step": 2, "action": "retrieve", "used": min(8, len(pairs))},
        {"step": 3, "action": "author", "chars": len(authored)},
        {"step": 4, "action": "synthesize_wiki", "title": wiki_title, "len": len(wiki_content)},
        {"step": 5, "action": "critic", "confidence": 0.82 if len(authored) > 260 else 0.60,
         "notes": ["Tighten examples for specificity.", "Link to issues/MRs/docs where possible."]},
    ]
    return {
        "answer": authored,
        "trace": trace,
        "applied_actions": applied,
        "wiki_title": wiki_title,
        "wiki_content": wiki_content,
        "wiki_path": created_path
    }

if nav == "Agent":
    st.markdown("**Plans the task, retrieves evidence, authors a clean example-rich note, and can create a wiki page (optional).**")
    msg = st.text_area("Goal or task", height=110, placeholder="e.g., Create an example-rich note on how values influence performance reviews. If gaps exist, propose a wiki draft.")
    auto = st.checkbox("Allow auto-actions (create wiki draft + incremental reindex)", value=False)

    if st.button("Run agent", type="primary"):
        if not msg.strip():
            st.warning("Please describe what you want the agent to do.")
        else:
            st.session_state["eka_busy"] = True
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-warm'>Loading</span></div>", unsafe_allow_html=True)
            res = agent_run(msg, auto_actions=auto)
            st.session_state["eka_busy"] = False
            _status_slot.markdown("<div style='display:flex; justify-content:flex-end'><span class='badge badge-ok'>Online</span></div>", unsafe_allow_html=True)

            st.subheader("Agent answer")
            st.write(res.get("answer","No answer generated."))

            st.markdown("#### What happened")
            for step in res.get("trace", []):
                if step["action"] == "rewrite_query":
                    st.write(f"• Rewrote to: `{step.get('output','')}`")
                elif step["action"] == "retrieve":
                    st.write(f"• Retrieved and used top {step.get('used',0)} chunks for authoring.")
                elif step["action"] == "author":
                    st.write(f"• Authored a concise answer ({step.get('chars',0)} chars).")
                elif step["action"] == "synthesize_wiki":
                    st.write(f"• Prepared wiki draft: **{step.get('title')}** ({step.get('len')} chars).")
                elif step["action"] == "critic":
                    st.write(f"• Critic confidence: {step.get('confidence',0.0):.2f}")
                    for n in step.get("notes", []):
                        st.write(f"  – {n}")

            if res.get("applied_actions"):
                st.success(f"Wiki page created and indexed: {res['applied_actions']}")
                with st.expander("Show created wiki page", expanded=True):
                    st.markdown(f"**Title:** {res.get('wiki_title','')}")
                    st.code(res.get("wiki_content",""), language="markdown")

# ===== UPLOAD =====
if nav == "Upload":
    st.markdown("**Add plain-text or Markdown files to your knowledge base. They will be cited in answers.**")
    f = st.file_uploader("Upload a file", type=["md","txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload & update knowledge"):
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
if nav == "About":
    st.markdown("### How this works")
    st.markdown("""
- **Hybrid retrieval**: semantic + (placeholder keyword) for recall & precision.
- **Reranker (optional)**: cross-encoder refines the top candidates.
- **Q&A**: answers strictly from retrieved passages; cites sources.
- **Agent authoring**: GPT writes a clean, example-rich note from top evidence; deterministic template fallback.
- **Incremental indexing**: uploads append vectors instead of full rebuilds.
- **Prebuilt index**: drop files into `data/index/prebuilt/` for instant startup.
""")
