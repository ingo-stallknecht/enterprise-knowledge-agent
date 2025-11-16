# app/streamlit_app.py
# Streamlit-only Enterprise Knowledge Agent with GPT-4 wiki drafting, safety filter,
# guarded deletes/edits (wiki folder only), and incremental indexing.
#
# Agent behaviors (confirmation required):
#   1) Create wiki pages (only if clearly requested)
#   2) Delete wiki pages (only if clearly requested; restricted to wiki folder)
#   3) Edit wiki pages (only if clearly requested; restricted to wiki folder)
#   4) Otherwise answer helpfully (no side effects)
#
# Safety:
#   - Harmful titles/content are blocked before creation/edit.
#   - Optional read-only demo mode: EKA_READ_ONLY="true" (disables create/delete/edit).

import sys, os, pathlib, re, time, tempfile, shutil, json, hashlib
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

# For direct GPT calls (wiki drafting)
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

def _secret(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)

# ---------- Writable caches (Streamlit Cloud friendly) ----------
WRITABLE_BASE = pathlib.Path(_secret("EKA_CACHE_DIR", "/mount/tmp")).expanduser()
if not WRITABLE_BASE.exists():
    WRITABLE_BASE = pathlib.Path(tempfile.gettempdir()) / "eka"
HF_CACHE = WRITABLE_BASE / "hf"
for p in (WRITABLE_BASE, HF_CACHE, WRITABLE_BASE / ".cache" / "huggingface", WRITABLE_BASE / ".huggingface"):
    p.mkdir(parents=True, exist_ok=True)
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

# ---------- OpenAI / LLM ----------
def _find_secret_key() -> str:
    candidates = ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_APIKEY", "openai_api_key", "openai_key"]
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
_key = _find_secret_key()
if _key:
    os.environ["OPENAI_API_KEY"] = _key

# Prefer GPT-4o for wiki drafting; respect user's model elsewhere
DEFAULT_LLM_MODEL = str(st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
os.environ["OPENAI_MODEL"] = DEFAULT_LLM_MODEL
os.environ["OPENAI_MAX_DAILY_USD"] = str(
    st.secrets.get("OPENAI_MAX_DAILY_USD", os.environ.get("OPENAI_MAX_DAILY_USD", "0.80"))
)

def _openai_diag() -> str:
    use = os.environ.get("USE_OPENAI", "false").lower() == "true"
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "")
    if use and key:
        return f"OpenAI: ON (key ✓, model {model})"
    if use and not key:
        return "OpenAI: ON (key missing)"
    return "OpenAI: OFF"

# ---------- Settings ----------
TOP_K = int(_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER_DEFAULT = _secret("EKA_USE_RERANKER", "true").lower() == "true"
DISABLE_RERANKER_BOOT = _secret("EKA_DISABLE_RERANKER", "false").lower() == "true"
AUTO_BOOTSTRAP = _secret("EKA_BOOTSTRAP", "true").lower() == "true"
USE_PREBUILT = _secret("EKA_USE_PREBUILT", "true").lower() == "true"

# Optional read-only (no UI toggle, for demos)
READ_ONLY = _secret("EKA_READ_ONLY", "false").lower() == "true"

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

# ---------- Config & dirs ----------
CFG = load_cfg("configs/settings.yaml") if pathlib.Path("configs/settings.yaml").exists() else {}
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
INDEX_DIR = pathlib.Path("data/index")

# Vector cache
VEC_DIR = INDEX_DIR / "vecs"
RECS_DIR = INDEX_DIR / "records"
MANIFEST = INDEX_DIR / "vecs_manifest.json"
for p in (VEC_DIR, RECS_DIR, MANIFEST.parent):
    p.mkdir(parents=True, exist_ok=True)

# ---------- CSS / UI ----------
st.set_page_config(page_title="Enterprise Knowledge Agent", page_icon="EKA", layout="wide")
st.markdown(
    """
<style>
.header-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.badge { border-radius: 999px; padding: 4px 10px; font-size: 0.85rem; border: 1px solid transparent; }
.badge-ok  { background:#E8FFF3; color:#05603A; border-color:#ABEFC6; }
.badge-err { background:#FFF1F1; color:#B42318; border-color:#FECDCA; }
.kpi{padding:8px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#FCFEFF;display:inline-block;margin-right:8px;}
.small{font-size:0.92rem;color:#687076;}
.cv-legend { display:flex; gap:10px; align-items:center; margin-bottom:6px; }
.cv-dot { width:10px; height:10px; display:inline-block; border-radius:999px; border:1px solid #d1d5db; }
.cv-b1 { background:#FEE2E2; } .cv-b2 { background:#FEF9C3; } .cv-b3 { background:#ECFEFF; } .cv-b4 { background:#F0FDF4; }
.cv-pill { display:inline-block; margin:4px 6px 8px 0; padding:6px 10px; border-radius:999px; border:1px solid #e5e7eb; font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Progress ----------
class Prog:
    def __init__(self, initial_text: str = "Working…"):
        self._pb = st.progress(0, text=initial_text)
        self._last = 0.0

    def update(self, label: Optional[str] = None, value: Optional[float] = None):
        if value is None:
            value = self._last
        value = max(0.0, min(float(value), 1.0))
        self._last = value
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

# ---------- Helpers: files, chunks, manifest ----------
def _scan_processed_files() -> List[pathlib.Path]:
    return sorted(PROC_DIR.glob("**/*.md"))

def _split_md(text: str) -> List[Dict]:
    return list(split_markdown(text, **CHUNK_CFG))

def have_any_markdown() -> bool:
    return any(PROC_DIR.rglob("*.md"))

def _hash_source(source_path: str) -> str:
    s = source_path.encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()[:16]

def load_manifest() -> Dict[str, Dict]:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(m: Dict[str, Dict]) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

def save_cache_for_source(source: str, vectors, records_list: List[Dict]) -> None:
    import numpy as np

    h = _hash_source(source)
    vpath = VEC_DIR / f"{h}.npy"
    rpath = RECS_DIR / f"{h}.json"
    np.save(vpath, vectors.astype("float32"))
    rpath.write_text(json.dumps(records_list, ensure_ascii=False, indent=0), encoding="utf-8")
    m = load_manifest()
    m[source] = {"vec": str(vpath), "recs": str(rpath), "n": len(records_list)}
    save_manifest(m)

def remove_cache_for_sources(sources: List[str]) -> List[str]:
    removed = []
    m = load_manifest()
    for s in sources:
        info = m.pop(s, None)
        if not info:
            continue
        for p in (info.get("vec"), info.get("recs")):
            if p:
                try:
                    pathlib.Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        removed.append(s)
    save_manifest(m)
    return removed

def concat_cache() -> Tuple[Optional["np.ndarray"], List[Dict]]:
    import numpy as np

    m = load_manifest()
    if not m:
        return None, []
    vecs, recs = [], []
    for _, info in m.items():
        v = info.get("vec")
        r = info.get("recs")
        if not (v and r):
            continue
        if not pathlib.Path(v).exists() or not pathlib.Path(r).exists():
            continue
        vecs.append(np.load(v))
        recs.extend(json.loads(pathlib.Path(r).read_text(encoding="utf-8")))
    if not vecs:
        return None, []
    X = np.vstack(vecs) if len(vecs) > 1 else vecs[0]
    return X.astype("float32"), recs

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
    return DocIndex(INDEX_PATH, STORE_PATH)

def save_index(X, records, progress: Optional[Prog] = None) -> None:
    idx = load_or_init_index()
    if progress:
        progress.update(label="Writing index…", value=0.92)
    idx.build(X, records)
    if progress:
        progress.update(label="Done", value=1.0)

def rebuild_from_cache(progress: Optional[Prog] = None) -> Dict:
    import numpy as np

    if progress:
        progress.update(label="Loading cached vectors…", value=0.1)
    X, records = concat_cache()
    if X is None:
        return rebuild_index(progress)
    if progress:
        progress.update(label=f"Assembling {len(records)} chunks…", value=0.35)
    save_index(X, records, progress)
    return {"num_files": len(set(r["source"] for r in records)), "num_chunks": len(records)}

def rebuild_index(progress: Optional[Prog] = None) -> Dict:
    files = _scan_processed_files()
    if progress:
        progress.update(label=f"Scanning… {len(files)} files", value=0.03)
    import numpy as np

    emb, _ = get_models()
    all_vecs, all_recs = [], []
    for i, fp in enumerate(files, 1):
        text = fp.read_text(encoding="utf-8")
        chunks = _split_md(text)
        recs = [{"text": c["text"], "source": str(fp).replace("\\", "/")} for c in chunks]
        texts = [c["text"] for c in chunks]
        if texts:
            V = emb.encode(texts)
            all_vecs.append(V)
            all_recs.extend(recs)
            save_cache_for_source(str(fp).replace("\\", "/"), V, recs)
        if progress:
            progress.update(
                label=f"Embedding {i}/{len(files)} files…", value=0.05 + 0.80 * (i / max(1, len(files)))
            )
    if all_vecs:
        X = np.vstack(all_vecs)
    else:
        import numpy as np

        X = np.zeros((0, 384), dtype="float32")
    save_index(X, all_recs, progress)
    return {"num_files": len(files), "num_chunks": len(all_recs)}

def incremental_add(markdown_text: str, source_path: str, progress: Optional[Prog] = None) -> Dict:
    chunks = _split_md(markdown_text)
    if progress:
        progress.update(label=f"Chunking {len(chunks)} chunks…", value=0.12)
    texts = [c["text"] for c in chunks]
    if not texts:
        return {"added_chunks": 0}
    emb, _ = get_models()
    V = emb.encode(texts)
    recs = [{"text": t, "source": source_path} for t in texts]
    save_cache_for_source(source_path, V, recs)
    idx = load_or_init_index()
    if hasattr(idx, "add"):
        if progress:
            progress.update(label="Appending to index…", value=0.7)
        idx.add(V.astype("float32"), recs)
        if progress:
            progress.update(label="Saved.", value=1.0)
        return {"added_chunks": len(texts), "appended": True}
    else:
        if progress:
            progress.update(label="Refreshing index from cache…", value=0.7)
        stats = rebuild_from_cache(progress)
        return {"added_chunks": len(texts), "appended": False, "rebuilt": stats}

# ---------- Retrieval & attribution ----------
def retrieve(query: str, k: int, mode: str, use_reranker: bool):
    t0 = time.time()
    emb, rer = get_models()
    qv = emb.encode([query])
    idx = load_or_init_index()
    if mode == "dense":
        raw = idx.query_dense(qv, k=max(k, 30))
        meta_mode = "dense"
    elif mode == "sparse":
        raw = idx.query_sparse(query, k=max(k, 30))
        meta_mode = "sparse"
    else:
        raw = idx.query_hybrid(qv, query, k=max(k, 30), alpha=ALPHA)
        meta_mode = "hybrid"
    candidates = [r for r, _ in raw]
    reranked, rerank_used = raw, False
    if use_reranker and rer is not None and candidates:
        pairs = rer.rerank(query, candidates[:30], top_k=max(k, 1))
        reranked = [(rec, sc) for rec, sc in pairs]
        rerank_used = True
    top_records = [r for r, _ in reranked[:k]]
    meta = {
        "mode": meta_mode,
        "alpha": ALPHA,
        "rerank_used": rerank_used,
        "retrieval_ms": int((time.time() - t0) * 1000),
    }
    return top_records, reranked, meta

_SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z0-9])")

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
        if len(p) < 40 and buf:
            buf = f"{buf} {p}"
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def attribute(answer: str, records: List[Dict]) -> List[dict]:
    if not answer or not records:
        return []
    emb, _ = get_models()
    sents = _split_sentences(answer)
    if not sents:
        return []
    svecs = emb.encode(sents)
    rvecs = emb.encode([(r.get("text") or "") for r in records])
    import numpy as np

    sim = (svecs @ rvecs.T).astype(float)
    out = []
    for i, s in enumerate(sents):
        j = int(np.argmax(sim[i]))
        best = records[j]
        out.append(
            {
                "sentence": s,
                "best_source": best.get("source"),
                "best_rank": j + 1,
                "score": float(sim[i, j]),
                "preview": (best.get("text") or "")[:240]
                + ("…" if len((best.get("text") or "")) > 240 else ""),
            }
        )
    return out

def fmt_citations(pairs: List[Tuple[Dict, float]], k: int) -> List[dict]:
    cites = []
    for i, (r, sc) in enumerate(pairs[:k], 1):
        txt = (r.get("text") or "").strip()
        cites.append(
            {
                "rank": i,
                "source": r.get("source"),
                "score": float(sc),
                "preview": (txt[:280] + "…") if len(txt) > 280 else txt,
            }
        )
    return cites

# ---------- Header ----------
def _index_health() -> Tuple[str, str]:
    try:
        idx = load_or_init_index()
        ok = (idx.size() > 0) if hasattr(idx, "size") else True
        if ok:
            return "<span class='badge badge-ok'>Online</span>", "ok"
        return "<span class='badge badge-err'>Empty</span>", "err"
    except Exception:
        return "<span class='badge badge-err'>Offline</span>", "err"

st.markdown(
    "<div class='header-row'>"
    "<div><h3>Enterprise Knowledge Agent</h3>"
    f"<div class='small'>Ask in plain language. Agent plans, retrieves, answers, cites. {_openai_diag()}</div></div>"
    "<div id='status-slot'></div></div>",
    unsafe_allow_html=True,
)
_status_slot = st.empty()
badge_html, _ = _index_health()
_status_slot.markdown(
    f"<div style='display:flex; justify-content:flex-end'>{badge_html}</div>",
    unsafe_allow_html=True,
)

# ---------- Orientation text for new users ----------
st.markdown(
    """
> **How to use this app**
>
> - **Ask** - Ask natural-language questions. The app searches a GitLab-inspired handbook plus your wiki pages  
>   and answers strictly from those documents with citations and a context visualizer.
> - **Agent** - Describe a task. The agent can  
>   1) propose and create wiki pages  
>   2) delete pages (only inside `data/processed/wiki/`)  
>   3) edit existing wiki pages  
>   4) answer normally with no side effects  
>
> **All create/delete/edit actions require your confirmation.**
>
> - **Upload** - Add your own `.md` / `.txt` files; they become part of the knowledge base and show up in answers.
> - **About** - See an overview of how the system works and browse the Markdown documents that are currently indexed.
>
> The initial corpus is based on a small subset of the public GitLab Handbook  
> (<https://about.gitlab.com/handbook/>), plus any wiki pages and uploads you add here.
"""
)

# ---------- Bootstrap (optional) ----------
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

def bootstrap_gitlab(progress: Optional[Prog] = None) -> Dict:
    if requests is None or md is None:
        return {"ok": False, "error": "requests/markdownify not available"}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    total, ok = len(CURATED), 0
    for i, url in enumerate(CURATED, 1):
        try:
            if progress:
                progress.update(
                    label=f"Fetching {i}/{total}: {url}", value=min(0.15, i / total * 0.15)
                )
            r = requests.get(url, headers={"User-Agent": "EKA-Streamlit/1.0"}, timeout=20)
            r.raise_for_status()
            html = r.text
            slug = url.rstrip("/").split("/")[-1] or "index"
            (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
            (PROC_DIR / f"{slug}.md").write_text(md(html), encoding="utf-8")
            ok += 1
            time.sleep(0.02)
        except Exception:
            pass
    return {"ok": ok > 0, "downloaded": ok, "total": len(CURATED)}

def corpus_stats() -> Dict:
    files = list(PROC_DIR.rglob("*.md"))
    n_files = len(files)
    n_bytes = sum((f.stat().st_size for f in files), 0)
    return {"files": n_files, "size_kb": int(n_bytes / 1024)}

def copy_prebuilt_if_available() -> bool:
    pre_faiss = PREBUILT_DIR / pathlib.Path(INDEX_PATH).name
    pre_store = PREBUILT_DIR / pathlib.Path(STORE_PATH).name
    if pre_faiss.exists() and pre_store.exists():
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        shutil.copy2(pre_faiss, INDEX_PATH)
        shutil.copy2(pre_store, STORE_PATH)
        return True
    return False

def ensure_ready_and_index(force_rebuild: bool = False) -> Dict:
    if USE_PREBUILT and pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists():
        return {"seeded": False, "rebuilt": False, "stats": corpus_stats()}
    if force_rebuild:
        prog = Prog("Rebuilding index from cache…")
        return {"seeded": False, "rebuilt": True, "stats": rebuild_from_cache(prog)}
    if not have_any_markdown():
        write_seed_corpus()
        prog = Prog("Indexing seed corpus…")
        stats = rebuild_index(prog)
        return {"seeded": True, "rebuilt": True, "stats": stats}
    if not pathlib.Path(INDEX_PATH).exists() or not pathlib.Path(STORE_PATH).exists():
        prog = Prog("Restoring index from cache…")
        stats = rebuild_from_cache(prog)
        return {"seeded": False, "rebuilt": True, "stats": stats}
    return {"seeded": False, "rebuilt": False, "stats": corpus_stats()}

if not (pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists()):
    if not (USE_PREBUILT and copy_prebuilt_if_available()):
        if AUTO_BOOTSTRAP and not have_any_markdown():
            st.info("Fetching a small subset of the GitLab Handbook…")
            res = bootstrap_gitlab(Prog("Bootstrapping…"))
            if not res.get("ok"):
                write_seed_corpus()
        ensure_ready_and_index(force_rebuild=False)

# ---------- Tabs ----------
tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

# ===== ASK =====
with tab_ask:
    st.markdown(
        "**Finds relevant passages and answers strictly from them. "
        "Shows citations and a sentence-to-source map.**"
    )
    q = st.text_area(
        "Your question",
        height=100,
        placeholder="For example: How are values applied in performance reviews?",
    )
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50)
    c1, c2 = st.columns(2)
    mode = c1.selectbox(
        "Retrieval mode",
        ["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(RETRIEVAL_MODE),
    )
    use_rr = c2.checkbox("Use reranker", value=USE_RERANKER_DEFAULT)
    if st.button("Get answer", type="primary", key="ask_btn"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            ensure_ready_and_index(False)
            recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)
            if len(pairs) == 0:
                ensure_ready_and_index(True)
                recs, pairs, meta = retrieve(q, TOP_K, mode, use_rr)
            ans, llm_meta = generate_answer(recs, q, max_chars=max_chars)
            if not ans or ans.strip() in {".", ""}:
                joined = "\n\n".join((r.get("text") or "") for r, _ in pairs[:6]).strip()
                ans = joined[:max_chars] if joined else "No relevant context found in the current corpus."
            st.subheader("Answer")
            st.write(ans)
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(
                f"<div class='kpi'>Sources: <b>{len(pairs[:TOP_K])}</b></div>",
                unsafe_allow_html=True,
            )
            m2.markdown(
                f"<div class='kpi'>Retrieval: <b>{meta.get('mode','')}</b></div>",
                unsafe_allow_html=True,
            )
            m3.markdown(
                f"<div class='kpi'>Latency: <b>{meta.get('retrieval_ms',0)} ms</b></div>",
                unsafe_allow_html=True,
            )
            m4.markdown(
                f"<div class='kpi'>LLM: <b>{llm_meta.get('llm','extractive')}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown("##### Why this answer: Context Visualizer")
            st.markdown(
                "<div class='cv-legend small'><span>Evidence strength:</span>"
                "<span class='cv-dot cv-b1'></span><span class='small'>weak</span>"
                "<span class='cv-dot cv-b2'></span><span class='small'>medium</span>"
                "<span class='cv-dot cv-b3'></span><span class='small'>strong</span>"
                "<span class='cv-dot cv-b4'></span><span class='small'>very strong</span></div>",
                unsafe_allow_html=True,
            )
            def band(s): return 4 if s >= 0.75 else 3 if s >= 0.50 else 2 if s >= 0.30 else 1
            def bcls(b): return {1: "cv-b1", 2: "cv-b2", 3: "cv-b3", 4: "cv-b4"}[b]
            pills = []
            for row in attribute(ans, recs):
                s = float(row.get("score", 0))
                sent = (row.get("sentence", "") or "").strip()
                src = row.get("best_source", "")
                rk = row.get("best_rank", "-")
                title = f"Top source #{rk}\n{src}\n(score {s:.2f})"
                pills.append(
                    f"<span class='cv-pill {bcls(band(s))}' title='{title}'>{sent}</span>"
                )
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.markdown("#### Sources")
            for c in fmt_citations(pairs, TOP_K):
                lab = f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}"
                with st.expander(lab, expanded=False):
                    st.write(c["preview"])

# ===== AGENT =====

# Session state for agent
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "mode": "idle",             # idle | create_proposed | delete_proposed | edit_proposed
        "proposed_title": "",
        "proposed_content": "",
        "delete_candidates": [],
        "edit_candidates": [],
        "edit_selected": "",
        "edit_original": "",
        "last_message": "",
        "last_display": "",
    }

# Intent regexes (DELETE first, then CREATE, then EDIT)
DELETE_PAT = re.compile(r"^\s*(delete|remove|trash|erase|drop)\b\s+(?P<target>.+)$", re.I)
CREATE_PAT = re.compile(r"\b(create|write|make|add|draft)\b.*\b(wiki|page|article|doc)\b", re.I)
EDIT_PAT   = re.compile(r"^\s*(edit|update|modify|change)\b\s+(?P<target>.+)$", re.I)

def classify_intent(msg: str) -> Tuple[str, Dict]:
    text = (msg or "").strip()

    # 1) Delete
    mdm = DELETE_PAT.search(text)
    if mdm:
        target = mdm.group("target").strip().strip('"\'')

        target = target.rstrip(".")
        return "delete_wiki", {"query": target[:200] if target else ""}

    # 2) Create (requires "create/write ... wiki/page/article/doc")
    if CREATE_PAT.search(text):
        m_title = re.search(r"['\"]([^'\"]{3,120})['\"]", text)
        if m_title:
            title = m_title.group(1)
        else:
            title = re.sub(
                r".*\b(wiki|page|article|doc)\b( about| on|:)?\s*",
                "",
                text,
                flags=re.I,
            )
        title = (title or "").strip().rstrip(".")
        return "create_wiki", {"title": title[:120] if title else "Untitled"}

    # 3) Edit
    em = EDIT_PAT.search(text)
    if em:
        target = em.group("target").strip().strip('"\'')

        target = target.rstrip(".")
        return "edit_wiki", {"query": target[:200] if target else ""}

    # 4) Answer normally
    return "answer", {}

def concise_title(title: str) -> str:
    t = (title or "Untitled").strip()
    t = re.sub(r"\?$", "", t)
    t = re.sub(r"^(create|write|make|draft|add)\s+", "", t, flags=re.I)
    if len(t) > 60:
        t = t[:57].rstrip() + "…"

    def _tc_word(w: str) -> str:
        if w.lower() in {"in", "and", "or", "of", "to", "for", "with", "on"}:
            return w.lower()
        return w.capitalize()

    # Avoid em dash; treat hyphen as separator
    parts = re.split(r"(\s+|\-)", t)
    t = "".join(_tc_word(w) if not w.isspace() and w != "-" else w for w in parts)
    t = re.sub(r"\s*-\s*", " - ", t)
    return t

# ---------- Safety filters ----------
_BLOCK_PATTERNS = [
    r"\b(?:kill|suicide|self\-harm|cutting)\b",
    r"\b(?:rape|sexual\s*violence|child\s*sexual|underage\s*sex|cp\b)\b",
    r"\b(?:hate|slur|ethnic\s*cleansing|genocide)\b",
    r"\b(?:terrorist\s*manual|bomb\s*making|explosive\s*recipe)\b",
    r"\b(?:porn|explicit\s*sexual|bestiality)\b",
    r"(?:\b(?:fuck|cunt|nigger|fag|retard)\b)",
]
_BLOCK_RE = re.compile("|".join(_BLOCK_PATTERNS), re.I)

def is_safe_text(title: str, content: str) -> Tuple[bool, str]:
    if not title.strip():
        return False, "Title is empty."
    if len(title) > 120:
        return False, "Title too long."
    if _BLOCK_RE.search(title) or _BLOCK_RE.search(content or ""):
        return False, "Title or content appears to contain harmful or disallowed content."
    return True, ""

def render_wiki_md_template(title: str, key_points: List[str]) -> str:
    bullets = "\n".join(f"- {p}" for p in key_points) if key_points else "- Add specific behaviors and links to evidence."
    tldr = (
        key_points[0]
        if key_points
        else "Summarize the most important value-driven behaviors with examples."
    )
    return f"""# {title}

> Internal guide drafted by the Knowledge Agent. Please review before publishing.

## TL;DR
- {tldr}

## Key behaviors
{bullets}

## Examples
- Add one example per behavior. Link to issue, merge request, or document where this was demonstrated.
- Describe outcomes (metrics, customer impact, quality).

## Checklist
- Document decisions and feedback in writing.
- Time-box feedback windows and collect input before the decision deadline.
- Tie observations to values and measurable outcomes.
- Link to evidence (issues, merge requests, documents).

## References
(Synthesized from indexed documents.)
"""

def _split_sentences_for_points(txt: str) -> List[str]:
    out = []
    for s in _split_sentences(txt or ""):
        s2 = s.strip()
        if (
            50 <= len(s2) <= 240
            and re.match(r"^[A-Z0-9].*[\.!\?]$", s2)
            and "http" not in s2
        ):
            out.append(s2)
    return out

def extract_key_points(
    query: str, pairs: List[Tuple[Dict, float]], emb: Embedder, max_points: int = 5
) -> List[str]:
    if not pairs:
        return []
    qv = emb.encode([query])
    cands: List[str] = []
    for rec, _ in pairs[:24]:
        cands.extend(_split_sentences_for_points(rec.get("text") or ""))
    if not cands:
        return []
    sv = emb.encode(cands)
    import numpy as np

    sim = (sv @ qv.T).reshape(-1)
    idxs = np.argsort(-sim)[: max(10, max_points * 3)]
    seen, out = set(), []
    for i in idxs:
        k = re.sub(r"[^a-z0-9]+", " ", cands[i].lower()).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(cands[i])
    if not any("time-box" in p.lower() or "time box" in p.lower() for p in out):
        out.insert(0, "Time-box feedback windows and collect input before the decision deadline.")
    return out[:max_points]

# ---------- GPT-4 wiki drafting ----------
def gpt_generate_wiki_md(preferred_title: str, query: str, key_points: List[str]) -> Optional[str]:
    """Use GPT-4o if available; otherwise fallback to configured model; if no client/key, return None."""
    if not (
        _OpenAI
        and os.environ.get("OPENAI_API_KEY", "").strip()
        and os.environ.get("USE_OPENAI", "false") == "true"
    ):
        return None
    client = _OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    bullets = "\n".join(f"- {p}" for p in key_points) if key_points else ""
    system = (
        "You are a helpful documentation writer for an internal engineering handbook. "
        "You produce clean, concise, safe Markdown pages with: "
        "H1 title, TL;DR, Key behaviors (bulleted), Examples, Checklist, References. "
        "The page must be safe: no hate, sexual content, self-harm, or instructions for violence or illicit acts. "
        "No filler. Keep it practical. Do not mention that this was written by a model."
    )
    user = (
        f"Write a wiki page titled: '{preferred_title}'.\n\n"
        f"User request/context: {query}\n\n"
        f"Key points to incorporate (if relevant):\n{bullets}\n\n"
        "Ensure feedback is time-boxed and collected before deadlines.\n"
        "Return only the Markdown content of the page."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=900,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                max_tokens=900,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except Exception:
            return None

# ---------- Wiki actions ----------
def agent_apply_create_wiki(title: str, content: str) -> Dict:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9\-]+", "-", title.lower()).strip("-") or "page"
    dst = WIKI_DIR / f"{slug}.md"
    dst.write_text(content, encoding="utf-8")
    created_path = str(dst).replace("\\", "/")
    prog = Prog("Updating index…")
    res = incremental_add(content, created_path, progress=prog)
    return {"file": f"{slug}.md", "path": created_path, "added_chunks": res.get("added_chunks", 0)}

def _is_in_wiki_folder(p: pathlib.Path) -> bool:
    try:
        p.resolve().relative_to(WIKI_DIR.resolve())
        return True
    except Exception:
        return False

def agent_apply_delete_wiki(selected_files: List[str]) -> Dict:
    deleted_paths: List[str] = []
    for f in selected_files:
        p = pathlib.Path(f)
        if not _is_in_wiki_folder(p):
            continue
        try:
            if p.exists():
                p.unlink()
                deleted_paths.append(str(p).replace("\\", "/"))
        except Exception:
            pass
    _ = remove_cache_for_sources(deleted_paths)
    prog = Prog("Refreshing index from cache…")
    _ = rebuild_from_cache(prog)
    return {"deleted": deleted_paths, "num_deleted": len(deleted_paths)}

def agent_apply_edit_wiki(path_str: str, new_content: str) -> Dict:
    """Edit an existing wiki file, keeping it in the wiki folder, and refresh its vectors only."""
    p = pathlib.Path(path_str)
    if not _is_in_wiki_folder(p):
        return {"edited": False, "error": "File not in wiki folder."}
    if not p.exists():
        return {"edited": False, "error": "File no longer exists."}

    p.write_text(new_content, encoding="utf-8")
    abs_path = str(p.resolve()).replace("\\", "/")

    # Drop old vectors for that source and rebuild index from remaining cache
    removed = remove_cache_for_sources([abs_path])
    prog = Prog("Updating index…")
    _ = rebuild_from_cache(prog)
    # Add fresh vectors for the edited file
    res_add = incremental_add(new_content, abs_path, progress=prog)
    return {
        "edited": True,
        "path": abs_path,
        "removed_old": removed,
        "added_chunks": res_add.get("added_chunks", 0),
    }

# ===== Agent tab =====
with tab_agent:
    st.markdown("**Agent**")
    st.write(
        "The agent can (1) propose and create wiki pages, (2) delete pages (wiki folder only), "
        "(3) edit wiki pages, or (4) answer normally. "
        "Create, delete, and edit actions always require your confirmation."
    )

    if READ_ONLY:
        st.info("Read-only mode is ON. Creating, deleting, and editing pages is disabled.")

    agent_result = st.container()
    state = st.session_state["agent_state"]

    # Main agent form (Run button stays near input)
    with st.form("agent_main_form", clear_on_submit=False):
        msg = st.text_area(
            "Goal or task",
            height=110,
            placeholder='Examples: create a wiki page "Values in Performance Reviews"; '
                        'delete values.md; edit values.md; or just ask a question.',
            key="agent_msg",
        )
        submitted_run = st.form_submit_button("Run", type="primary")

    if submitted_run:
        text = (msg or "").strip()
        if not text:
            with agent_result:
                st.warning("Please describe what you want the agent to do.")
        else:
            intent, info = classify_intent(text)

            if intent == "create_wiki":
                if READ_ONLY:
                    with agent_result:
                        st.error("Create denied: read-only mode is enabled.")
                else:
                    ensure_ready_and_index(False)
                    query = text if text.endswith("?") else text.rstrip(".") + "?"
                    recs, pairs, _m = retrieve(query, min(TOP_K, 12), "hybrid", use_reranker=False)
                    emb, _ = get_models()
                    points = extract_key_points(query, pairs, emb, max_points=5)
                    title = concise_title(info.get("title") or "Untitled")

                    draft = gpt_generate_wiki_md(title, query, points)
                    if draft is None or not draft.strip():
                        draft = render_wiki_md_template(title, points)

                    ok, why = is_safe_text(title, draft)
                    if not ok:
                        with agent_result:
                            st.error(f"Blocked by safety filter: {why}")
                    else:
                        state.update(
                            {
                                "mode": "create_proposed",
                                "proposed_title": title,
                                "proposed_content": draft,
                                "delete_candidates": [],
                                "edit_candidates": [],
                                "edit_selected": "",
                                "edit_original": "",
                                "last_message": text,
                            }
                        )

            elif intent == "delete_wiki":
                if READ_ONLY:
                    with agent_result:
                        st.error("Delete denied: read-only mode is enabled.")
                else:
                    q = (info.get("query") or "").lower()
                    cands = []
                    for p in WIKI_DIR.glob("*.md"):
                        if q in p.stem.lower() or q in p.name.lower():
                            cands.append(str(p).replace("\\", "/"))
                    state.update(
                        {
                            "mode": "delete_proposed",
                            "delete_candidates": cands,
                            "proposed_title": "",
                            "proposed_content": "",
                            "edit_candidates": [],
                            "edit_selected": "",
                            "edit_original": "",
                            "last_message": text,
                        }
                    )
                    if not cands:
                        with agent_result:
                            st.info("No matching wiki files found in the wiki folder.")

            elif intent == "edit_wiki":
                if READ_ONLY:
                    with agent_result:
                        st.error("Edit denied: read-only mode is enabled.")
                else:
                    q = (info.get("query") or "").lower()
                    cands = []
                    for p in WIKI_DIR.glob("*.md"):
                        if q in p.stem.lower() or q in p.name.lower():
                            cands.append(str(p).replace("\\", "/"))
                    if not cands:
                        state.update(
                            {
                                "mode": "edit_proposed",
                                "edit_candidates": [],
                                "edit_selected": "",
                                "edit_original": "",
                                "delete_candidates": [],
                                "proposed_title": "",
                                "proposed_content": "",
                                "last_message": text,
                            }
                        )
                        with agent_result:
                            st.info("No matching wiki files found in the wiki folder.")
                    else:
                        # Let user pick which file to edit in a separate form below
                        state.update(
                            {
                                "mode": "edit_proposed",
                                "edit_candidates": cands,
                                "edit_selected": cands[0],
                                "edit_original": pathlib.Path(cands[0]).read_text(
                                    encoding="utf-8", errors="ignore"
                                ),
                                "delete_candidates": [],
                                "proposed_title": "",
                                "proposed_content": "",
                                "last_message": text,
                            }
                        )

            else:
                # Helpful answer (no side effects)
                ensure_ready_and_index(False)
                recs, pairs, meta = retrieve(text, TOP_K, RETRIEVAL_MODE, USE_RERANKER_DEFAULT)
                if len(pairs) == 0:
                    ensure_ready_and_index(True)
                    recs, pairs, meta = retrieve(text, TOP_K, RETRIEVAL_MODE, USE_RERANKER_DEFAULT)
                ans, llm_meta = generate_answer(recs, text, max_chars=900)
                if not ans or ans.strip() in {".", ""}:
                    joined = "\n\n".join((r.get("text") or "") for r, _ in pairs[:6]).strip()
                    ans = joined[:900] if joined else "No relevant context found in the current corpus."
                state.update(
                    {
                        "mode": "idle",
                        "last_display": ans,
                    }
                )
                with agent_result:
                    st.subheader("Agent answer")
                    st.write(ans)
                    st.markdown("#### Sources")
                    for c in fmt_citations(pairs, k=min(6, len(pairs))):
                        label = f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}"
                        with st.expander(label, expanded=False):
                            st.write(c["preview"])

    # CREATE confirmation
    if state["mode"] == "create_proposed":
        st.subheader("Create wiki - confirmation required")
        with st.form("agent_create_confirm", clear_on_submit=False):
            title_val = st.text_input(
                "Title",
                value=state["proposed_title"],
                max_chars=120,
            )
            st.code(state["proposed_content"], language="markdown")
            ok2, why2 = is_safe_text(title_val, state["proposed_content"])
            if not ok2:
                st.error(f"Blocked by safety filter: {why2}")
            c1, c2 = st.columns(2)
            confirm_create = c1.form_submit_button(
                "Confirm create page",
                disabled=not ok2 or READ_ONLY,
            )
            cancel_create = c2.form_submit_button("Cancel")
        if confirm_create:
            if READ_ONLY:
                st.error("Create denied: read-only mode is enabled.")
            else:
                res = agent_apply_create_wiki(
                    title_val.strip() or state["proposed_title"],
                    state["proposed_content"],
                )
                st.success(f"Created and indexed: {res['file']}")
                if res.get("path"):
                    st.caption(res["path"])
                st.session_state["agent_state"] = {
                    "mode": "idle",
                    "proposed_title": "",
                    "proposed_content": "",
                    "delete_candidates": [],
                    "edit_candidates": [],
                    "edit_selected": "",
                    "edit_original": "",
                    "last_message": "",
                    "last_display": f"Created {res['file']}",
                }
        elif cancel_create:
            st.session_state["agent_state"]["mode"] = "idle"
            st.info("Create action canceled.")

    # DELETE confirmation
    if state["mode"] == "delete_proposed":
        st.subheader("Delete wiki - confirmation required (wiki folder only)")
        if not state["delete_candidates"]:
            with st.form("agent_delete_close"):
                close_btn = st.form_submit_button("Close")
            if close_btn:
                st.session_state["agent_state"]["mode"] = "idle"
        else:
            with st.form("agent_delete_confirm", clear_on_submit=False):
                sel = st.multiselect(
                    "Select files to delete (restricted to data/processed/wiki/)",
                    options=state["delete_candidates"],
                    default=state["delete_candidates"][:1],
                )
                d1, d2 = st.columns(2)
                confirm_delete = d1.form_submit_button(
                    "Confirm delete selected",
                    disabled=READ_ONLY or not sel,
                )
                cancel_delete = d2.form_submit_button("Cancel")
            if confirm_delete:
                if READ_ONLY:
                    st.error("Delete denied: read-only mode is enabled.")
                else:
                    res = agent_apply_delete_wiki(sel)
                    st.success(f"Deleted {res['num_deleted']} file(s).")
                    if res.get("deleted"):
                        st.code("\n".join(res["deleted"]), language="text")
                    st.session_state["agent_state"] = {
                        "mode": "idle",
                        "proposed_title": "",
                        "proposed_content": "",
                        "delete_candidates": [],
                        "edit_candidates": [],
                        "edit_selected": "",
                        "edit_original": "",
                        "last_message": "",
                        "last_display": f"Deleted {res['num_deleted']} file(s)",
                    }
            elif cancel_delete:
                st.session_state["agent_state"]["mode"] = "idle"
                st.info("Delete action canceled.")

    # EDIT confirmation
    if state["mode"] == "edit_proposed":
        st.subheader("Edit wiki - confirmation required (wiki folder only)")
        if not state["edit_candidates"]:
            with st.form("agent_edit_close"):
                close_btn = st.form_submit_button("Close")
            if close_btn:
                st.session_state["agent_state"]["mode"] = "idle"
        else:
            with st.form("agent_edit_confirm", clear_on_submit=False):
                # Select file to edit
                file_choice = st.selectbox(
                    "Select file to edit (restricted to data/processed/wiki/)",
                    options=state["edit_candidates"],
                    index=max(0, state["edit_candidates"].index(state["edit_selected"]))
                    if state["edit_selected"] in state["edit_candidates"]
                    else 0,
                )
                # Load content if changed
                if file_choice != state["edit_selected"] or not state["edit_original"]:
                    try:
                        content_now = pathlib.Path(file_choice).read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except Exception:
                        content_now = ""
                    state["edit_selected"] = file_choice
                    state["edit_original"] = content_now

                edited_text = st.text_area(
                    "Content",
                    value=state["edit_original"],
                    height=260,
                )
                file_title = pathlib.Path(state["edit_selected"]).name
                ok3, why3 = is_safe_text(file_title, edited_text)
                if not ok3:
                    st.error(f"Blocked by safety filter: {why3}")
                e1, e2 = st.columns(2)
                confirm_edit = e1.form_submit_button(
                    "Confirm save changes",
                    disabled=READ_ONLY or not ok3,
                )
                cancel_edit = e2.form_submit_button("Cancel")
            if confirm_edit:
                if READ_ONLY:
                    st.error("Edit denied: read-only mode is enabled.")
                else:
                    res = agent_apply_edit_wiki(state["edit_selected"], edited_text)
                    if not res.get("edited"):
                        st.error(f"Edit failed: {res.get('error','unknown error')}")
                    else:
                        st.success("Changes saved and index updated.")
                        st.caption(res.get("path", ""))
                        st.session_state["agent_state"] = {
                            "mode": "idle",
                            "proposed_title": "",
                            "proposed_content": "",
                            "delete_candidates": [],
                            "edit_candidates": [],
                            "edit_selected": "",
                            "edit_original": "",
                            "last_message": "",
                            "last_display": f"Edited {pathlib.Path(res.get('path','')).name}",
                        }
            elif cancel_edit:
                st.session_state["agent_state"]["mode"] = "idle"
                st.info("Edit action canceled.")

# ===== UPLOAD =====
with tab_upload:
    st.markdown(
        "**Add plain-text or Markdown files to your knowledge base. They will be cited in answers.**"
    )
    f = st.file_uploader("Upload a file", type=["md", "txt"])
    ttl = st.text_input("Optional page title")
    if st.button("Upload and update knowledge", key="upload_btn"):
        if not f:
            st.warning("Please select a file first.")
        else:
            name = (ttl or f.name).strip() or "page"
            slug = re.sub(r"[^a-z0-9\-]+", "-", name.lower()).strip("-") or "page"
            WIKI_DIR.mkdir(parents=True, exist_ok=True)
            text = f.getvalue().decode("utf-8", errors="ignore")
            dst = WIKI_DIR / f"{slug}.md"
            dst.write_text(text, encoding="utf-8")

            prog = Prog("Updating index…")
            res = incremental_add(text, str(dst).replace("\\", "/"), progress=prog)
            if res.get("appended", False):
                st.success(f"✅ Incrementally indexed {res['added_chunks']} chunks from {slug}.md")
            else:
                st.success(
                    f"✅ Refreshed index from cache; added {res['added_chunks']} chunks."
                )

# ===== ABOUT =====
def _list_md_files() -> List[Dict]:
    files = sorted(PROC_DIR.rglob("*.md"))
    out = []
    MAX_CHARS = 2000  # show full content for small files, truncate very long ones
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        full_text = text.strip()
        truncated = False
        if len(full_text) > MAX_CHARS:
            # Cut at a line boundary near MAX_CHARS if possible
            snippet = full_text[:MAX_CHARS]
            last_newline = snippet.rfind("\n")
            if last_newline > MAX_CHARS * 0.7:
                snippet = snippet[:last_newline]
            preview = snippet.strip()
            truncated = True
        else:
            preview = full_text
        out.append(
            {
                "path": str(f).replace("\\", "/"),
                "name": f.name,
                "kb": max(1, f.stat().st_size // 1024),
                "preview": preview,
                "truncated": truncated,
            }
        )
    return out

with tab_about:
    st.markdown("### How this works")
    st.markdown(
        """
- Hybrid retrieval with optional reranker.
- Question answering strictly from retrieved passages with citations.
- Agent can create, delete, and edit wiki pages in `data/processed/wiki/` (always with confirmation), or just answer.
- Vector cache so deletes and edits do not re-embed all files; the index is rebuilt from cached vectors.
- Prebuilt index support via `data/index/prebuilt/` for faster cold starts.
- Safety filter on titles and content; destructive actions are restricted to the wiki folder.
- Documents view below where you can inspect the Markdown files that are currently part of the corpus.
"""
    )
    stats_now = corpus_stats()
    st.info(f"Corpus stats: {stats_now['files']} files, approximately {stats_now['size_kb']} KB")

    st.markdown("#### Current Markdown files")
    md_list = _list_md_files()
    if not md_list:
        st.write("No Markdown files found yet.")
    else:
        for item in md_list:
            header = f"{item['name']}  ·  approximately {item['kb']} KB  ·  {item['path']}"
            with st.expander(header, expanded=False):
                if item.get("preview"):
                    st.code(item["preview"], language="markdown")
                    if item.get("truncated"):
                        st.caption("Preview truncated. The file is longer in full.")
                else:
                    st.write("(empty)")
