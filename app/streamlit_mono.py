# app/streamlit_mono.py
# Streamlit-only Enterprise Knowledge Agent with true incremental delete (no full re-embed)
# and cleaned Agent UI state to avoid odd reruns/tab jumps.

import sys, os, pathlib, re, time, tempfile, shutil
from typing import List, Dict, Tuple, Optional
import streamlit as st

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

WRITABLE_BASE = pathlib.Path(_secret("EKA_CACHE_DIR", "/mount/tmp")).expanduser()
if not WRITABLE_BASE.exists():
    WRITABLE_BASE = pathlib.Path(tempfile.gettempdir()) / "eka"
for p in (WRITABLE_BASE, WRITABLE_BASE / "hf", WRITABLE_BASE / ".cache" / "huggingface", WRITABLE_BASE / ".huggingface"):
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
    "HF_HOME": str(WRITABLE_BASE / "hf"),
    "TRANSFORMERS_CACHE": str(WRITABLE_BASE / "hf"),
    "SENTENCE_TRANSFORMERS_HOME": str(WRITABLE_BASE / "hf"),
    "HUGGINGFACE_HUB_CACHE": str(WRITABLE_BASE / "hf"),
    "XDG_CACHE_HOME": str(WRITABLE_BASE),
})

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
if _key: os.environ["OPENAI_API_KEY"] = _key
os.environ["OPENAI_MODEL"] = str(st.secrets.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o")))
os.environ["OPENAI_MAX_DAILY_USD"] = str(st.secrets.get("OPENAI_MAX_DAILY_USD", os.environ.get("OPENAI_MAX_DAILY_USD", "0.80")))

def _openai_diag() -> str:
    use = os.environ.get("USE_OPENAI", "false").lower() == "true"
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "")
    if use and key: return f"OpenAI: ON (key ✓, model {model})"
    if use and not key: return "OpenAI: ON (key missing ⚠)"
    return "OpenAI: OFF"

TOP_K = int(_secret("EKA_TOP_K", "12"))
MAX_CHARS_DEFAULT = int(_secret("EKA_MAX_CHARS", "900"))
RETRIEVAL_MODE = _secret("EKA_RETRIEVAL_MODE", "hybrid")
USE_RERANKER_DEFAULT = _secret("EKA_USE_RERANKER", "true").lower() == "true"
DISABLE_RERANKER_BOOT = _secret("EKA_DISABLE_RERANKER", "false").lower() == "true"
AUTO_BOOTSTRAP = _secret("EKA_BOOTSTRAP", "true").lower() == "true"
USE_PREBUILT = _secret("EKA_USE_PREBUILT", "true").lower() == "true"

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.rag.utils import ensure_dirs, load_cfg
from app.rag.embedder import Embedder
from app.rag.index import DocIndex
from app.rag.reranker import Reranker
from app.rag.chunker import split_markdown
from app.llm.answerer import generate_answer

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

st.set_page_config(page_title="Enterprise Knowledge Agent", page_icon="✨", layout="wide")
st.markdown("""
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
""", unsafe_allow_html=True)

class Prog:
    def __init__(self, initial_text: str = "Working…"):
        self._pb = st.progress(0, text=initial_text); self._last = 0.0
    def update(self, label: Optional[str] = None, value: Optional[float] = None):
        if value is None: value = self._last
        value = max(0.0, min(float(value), 1.0)); self._last = value
        self._pb.progress(int(value * 100), text=label if label is not None else None)

SEED_MD = {
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

def copy_prebuilt_if_available() -> bool:
    pre_f = PREBUILT_DIR / pathlib.Path(INDEX_PATH).name
    pre_s = PREBUILT_DIR / pathlib.Path(STORE_PATH).name
    if pre_f.exists() and pre_s.exists():
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        shutil.copy2(pre_f, INDEX_PATH); shutil.copy2(pre_s, STORE_PATH)
        return True
    return False

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
    try: idx.load()
    except Exception: pass
    return idx

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
        X = np.zeros((0, 384), dtype="float32")
    idx = load_or_init_index()
    if progress: progress.update(label="Writing index…", value=0.9)
    idx.build(X, records)
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
    res = idx.add(vecs, records)
    if progress: progress.update(label="Saved.", value=1.0)
    return {"added_chunks": len(texts), **res}

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

def _openai_status_badge_html() -> str:
    return f"<span class='badge badge-ok'>Online</span>"

# ---------- header
st.markdown(
    "<div class='header-row'>"
    "<div><h3>Enterprise Knowledge Agent</h3>"
    f"<div class='small'>Ask in plain language. Agent plans, retrieves, answers, cites. {_openai_diag()}</div></div>"
    "<div id='status-slot'></div></div>",
    unsafe_allow_html=True
)
_status_slot = st.empty()
_status_slot.markdown(f"<div style='display:flex; justify-content:flex-end'>{_openai_status_badge_html()}</div>", unsafe_allow_html=True)

# ---------- bootstrap / readiness
def bootstrap_gitlab(progress: Optional[Prog] = None) -> Dict:
    if requests is None or md is None:
        return {"ok": False, "error": "requests/markdownify not available"}
    RAW_DIR.mkdir(parents=True, exist_ok=True); PROC_DIR.mkdir(parents=True, exist_ok=True)
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

def copy_or_seed_then_index():
    if USE_PREBUILT and copy_prebuilt_if_available():
        return
    if have_any_markdown() and pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists():
        return
    prog = Prog("Preparing…")
    did_any = False
    if AUTO_BOOTSTRAP:
        st.warning("No corpus found — attempting to fetch the GitLab Handbook subset.", icon="⚠️")
        fetch_res = bootstrap_gitlab(progress=prog)
        if fetch_res.get("ok"):
            _ = rebuild_index(prog)
            st.success(f"Fetched {fetch_res.get('downloaded',0)} pages and built index.", icon="✅")
            did_any = True
    if not did_any:
        st.info("Using built-in seed corpus.", icon="ℹ️")
        write_seed_corpus()
        _ = rebuild_index(prog)
        st.success("Seed corpus indexed. You can now ask questions.", icon="✅")

def ensure_ready_and_index(force_rebuild: bool = False) -> Dict:
    idx_files_ok = pathlib.Path(INDEX_PATH).exists() and pathlib.Path(STORE_PATH).exists()
    if not have_any_markdown():
        write_seed_corpus(); stats = rebuild_index(Prog("Indexing seed corpus…"))
        return {"seeded": True, "rebuilt": True, "stats": stats}
    if force_rebuild or not idx_files_ok:
        stats = rebuild_index(Prog("Rebuilding index…")); return {"seeded": False, "rebuilt": True, "stats": stats}
    # sanity probe
    try:
        emb, _ = get_models(); qv = emb.encode(["ping"]); idx = load_or_init_index()
        if not idx.query_dense(qv, k=1):
            stats = rebuild_index(Prog("Repairing empty index…")); return {"seeded": False, "rebuilt": True, "stats": stats}
    except Exception:
        stats = rebuild_index(Prog("Repairing index…")); return {"seeded": False, "rebuilt": True, "stats": stats}
    return {"seeded": False, "rebuilt": False, "stats": corpus_stats()}

copy_or_seed_then_index()
_ = ensure_ready_and_index(False)

# ---------- tabs
tab_ask, tab_agent, tab_upload, tab_about = st.tabs(["Ask", "Agent", "Upload", "About"])

# ===== ASK =====
with tab_ask:
    st.markdown("**Finds relevant passages and answers strictly from them. Shows citations and a sentence-to-source map.**")
    q = st.text_area("Your question", height=100, placeholder="e.g., How are values applied in performance reviews?", key="ask_q")
    max_chars = st.slider("Answer length limit", 200, 2000, MAX_CHARS_DEFAULT, 50, key="ask_len")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval mode", ["hybrid","dense","sparse"],
                        index=["hybrid","dense","sparse"].index(RETRIEVAL_MODE), key="ask_mode")
    use_rr = c2.checkbox("Use reranker", value=USE_RERANKER_DEFAULT, key="ask_rr")
    if st.button("Get answer", type="primary", key="ask_btn"):
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
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {"mode": "idle", "proposed_title": "", "proposed_content": "", "delete_candidates": [], "last_result": None}

CREATE_PAT = re.compile(r"\b(create|write|make|add|draft)\b.*\b(wiki|page|article|doc)\b", re.I)
DELETE_PAT = re.compile(r"\b(delete|remove|trash|erase|drop)\b\s+(.+)", re.I)

def classify_intent(msg: str) -> Tuple[str, Dict]:
    text = (msg or "").strip()
    if CREATE_PAT.search(text):
        m = re.search(r"['\"]([^'\"]{3,120})['\"]", text)
        title = m.group(1) if m else re.sub(r".*\b(wiki|page|article|doc)\b( about| on|:)?\s*", "", text, flags=re.I)
        title = (title or "").strip().rstrip(".")
        return "create_wiki", {"title": title[:120] if title else "Untitled"}
    mdm = DELETE_PAT.search(text)
    if mdm:
        target = mdm.group(2).strip().strip('"\'').rstrip(".")
        return "delete_wiki", {"query": target[:200] if target else ""}
    return "answer", {}

def concise_title(title: str) -> str:
    t = (title or "Untitled").strip()
    t = re.sub(r"\?$", "", t)
    t = re.sub(r"^(create|write|make|draft|add)\s+", "", t, flags=re.I)
    if len(t) > 60: t = t[:57].rstrip() + "…"
    def _tc(s):
        return " ".join(
            w.capitalize() if w.lower() not in {"in","and","or","of","to","for","with","on"}
            else w.lower()
            for w in re.split(r"(\s+|\—)", s)
        )
    t = _tc(t); t = re.sub(r"\s*—\s*", " — ", t)
    return t

def render_wiki_md(title: str, key_points: List[str]) -> str:
    bullets = "\n".join(f"- {p}" for p in key_points) if key_points else "- Add specific behaviors and links to evidence."
    tldr = key_points[0] if key_points else "Summarize the most important value-driven behaviors with examples."
    return f"""# {title}

> Internal guide – drafted by the Knowledge Agent. Please review before publishing.

## TL;DR
- {tldr}

## Key behaviors
{bullets}

## Examples
- Add one example per behavior. Link to issue/MR/doc where demonstrated.
- Describe outcomes (metrics, customer impact, quality).

## Checklist
- Document decisions and feedback in writing.
- Time-box feedback windows and collect input before the decision deadline.
- Tie observations to values and measurable outcomes.
- Link to evidence (issues/MRs/docs).

## References
(Agent synthesized from indexed documents.)
"""

def _split_sentences_for_points(txt: str) -> List[str]:
    out = []
    for s in _split_sentences(txt or ""):
        s2 = s.strip()
        if 50 <= len(s2) <= 240 and re.match(r"^[A-Z0-9].*[\.!\?]$", s2) and "http" not in s2:
            out.append(s2)
    return out

def extract_key_points(query: str, pairs: List[Tuple[Dict, float]], emb: Embedder, max_points: int = 5) -> List[str]:
    if not pairs: return []
    qv = emb.encode([query])
    cands = []
    for rec, _ in pairs[:24]:
        cands.extend(_split_sentences_for_points(rec.get("text") or ""))
    if not cands: return []
    sv = emb.encode(cands)
    import numpy as np
    sim = (sv @ qv.T).reshape(-1)
    idxs = np.argsort(-sim)[: max(10, max_points*3)]
    seen, out = set(), []
    for i in idxs:
        k = re.sub(r"[^a-z0-9]+", " ", cands[i].lower()).strip()
        if k and k not in seen:
            seen.add(k); out.append(cands[i])
    if not any("time-box" in p.lower() or "time box" in p.lower() for p in out):
        out.insert(0, "Time-box feedback windows and collect input before the decision deadline.")
    return out[:max_points]

def agent_apply_create_wiki(title: str, content: str) -> Dict:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9\-]+", "-", title.lower()).strip("-") or "page"
    dst = WIKI_DIR / f"{slug}.md"
    dst.write_text(content, encoding="utf-8")
    created_path = str(dst).replace("\\","/")
    prog = Prog("Updating index…")
    res = incremental_add(content, created_path, progress=prog)  # append only new file
    return {"file": f"{slug}.md", "path": created_path, "details": res}

def agent_apply_delete_wiki(selected_files: List[str]) -> Dict:
    idx = load_or_init_index()
    normalized = [str(pathlib.Path(f)).replace("\\","/") for f in selected_files or []]
    res = idx.remove_sources(normalized)  # targeted vector removal; no re-embedding
    return {"deleted": normalized, "details": res}

with tab_agent:
    st.markdown("**Agent**")
    st.write("The agent has three behaviors:\n"
             "1) If your request clearly asks to **create a wiki page**, it will propose a title & content and ask for confirmation.\n"
             "2) If your request asks to **delete a page**, it will list matching files and ask for confirmation.\n"
             "3) Otherwise, it will **answer helpfully** without side effects.\n"
             "Nothing happens without your approval.")
    state = st.session_state["agent_state"]

    # persistent result (no tab jumps)
    if state["last_result"]:
        r = state["last_result"]
        if r.get("action") == "create":
            st.success(f"Created and indexed: **{r['title']}**  → `{r['file']}`")
            st.caption(r.get("path", ""))
            with st.expander("Details"): st.json(r.get("details", {}))
        elif r.get("action") == "delete":
            st.success(f"Deleted **{len(r.get('deleted', []))}** file(s) from index (no re-embed).")
            if r.get("deleted"): st.code("\n".join(r["deleted"]), language="text")
            with st.expander("Details"): st.json(r.get("details", {}))

    if state["mode"] == "create_proposed":
        st.subheader("Create wiki — confirmation required")
        title_val = st.text_input("Title", value=state["proposed_title"], key="agent_create_title")
        st.code(state["proposed_content"], language="markdown")
        c1, c2 = st.columns(2)
        if c1.button("Confirm create page", key="agent_confirm_create"):
            res = agent_apply_create_wiki(title_val.strip() or state["proposed_title"], state["proposed_content"])
            st.success(f"Created and indexed: **{title_val.strip() or state['proposed_title']}**  → `{res['file']}`")
            st.caption(res.get("path", ""))
            with st.expander("Details"): st.json(res)
            st.session_state["agent_state"] = {
                "mode": "idle", "proposed_title": "", "proposed_content": "", "delete_candidates": [],
                "last_result": {"action": "create", "title": title_val.strip() or state["proposed_title"],
                                "file": res.get("file",""), "path": res.get("path",""), "details": res}
            }
        if c2.button("Cancel", key="agent_cancel_create"):
            st.session_state["agent_state"]["mode"] = "idle"

    elif state["mode"] == "delete_proposed":
        st.subheader("Delete wiki — confirmation required")
        if not state["delete_candidates"]:
            st.info("No matching wiki files found. Tip: use a distinctive part of the filename or title.")
            if st.button("Close", key="agent_close_delete_empty"):
                st.session_state["agent_state"]["mode"] = "idle"
        else:
            sel = st.multiselect("Select files to delete", options=state["delete_candidates"],
                                 default=state["delete_candidates"][:1], key="agent_delete_sel")
            d1, d2 = st.columns(2)
            if d1.button("Confirm delete selected", key="agent_confirm_delete"):
                res = agent_apply_delete_wiki(sel)
                st.success(f"Deleted **{len(res.get('deleted', []))}** file(s) from index (no re-embed).")
                if res.get("deleted"): st.code("\n".join(res["deleted"]), language="text")
                with st.expander("Details"): st.json(res)
                st.session_state["agent_state"] = {
                    "mode": "idle", "proposed_title": "", "proposed_content": "", "delete_candidates": [],
                    "last_result": {"action": "delete", "deleted": res.get("deleted", []), "details": res}
                }
            if d2.button("Cancel", key="agent_cancel_delete"):
                st.session_state["agent_state"]["mode"] = "idle"

    else:
        msg = st.text_area("Goal or task", height=110,
                           placeholder='e.g., create a wiki page "Values in Reviews" · delete values.md · or just ask a question',
                           key="agent_msg")
        if st.button("Run", type="primary", key="agent_run_btn"):
            text = (msg or "").strip()
            if not text:
                st.warning("Please describe what you want the agent to do.")
            else:
                intent, info = classify_intent(text)
                if intent == "create_wiki":
                    query = text if text.endswith("?") else (text.rstrip(".") + "?")
                    recs, pairs, _m = retrieve(query, min(TOP_K, 12), "hybrid", use_reranker=False)
                    emb, _ = get_models()
                    points = extract_key_points(query, pairs, emb, max_points=5)
                    title = concise_title(info.get("title") or "Untitled")
                    content = render_wiki_md(title, points)
                    st.session_state["agent_state"].update({
                        "mode": "create_proposed", "proposed_title": title,
                        "proposed_content": content, "last_result": None
                    })
                elif intent == "delete_wiki":
                    q = (info.get("query") or "").lower()
                    cands = []
                    for p in WIKI_DIR.glob("*.md"):
                        if q in p.stem.lower() or q in p.name.lower():
                            cands.append(str(p).replace("\\","/"))
                    st.session_state["agent_state"].update({
                        "mode": "delete_proposed", "delete_candidates": cands, "last_result": None
                    })
                else:
                    # helpful answer only
                    ensure_ready_and_index(False)
                    recs, pairs, meta = retrieve(text, TOP_K, RETRIEVAL_MODE, USE_RERANKER_DEFAULT)
                    if len(pairs) == 0:
                        ensure_ready_and_index(True)
                        recs, pairs, meta = retrieve(text, TOP_K, RETRIEVAL_MODE, USE_RERANKER_DEFAULT)
                    ans, llm_meta = generate_answer(recs, text, max_chars=900)
                    if not ans or ans.strip() in {".", ""}:
                        joined = "\n\n".join((r.get("text") or "") for r,_ in pairs[:6]).strip()
                        ans = joined[:900] if joined else "No relevant context found in the current corpus."
                    st.subheader("Agent answer"); st.write(ans)
                    st.markdown("#### Sources")
                    for c in fmt_citations(pairs, k=min(6, len(pairs))):
                        with st.expander(f"Source #{c['rank']} — {c['source']}  ·  score={c['score']:.3f}", expanded=False):
                            st.write(c["preview"])

# ===== UPLOAD =====
with tab_upload:
    st.markdown("**Add plain-text or Markdown files to your knowledge base. They will be cited in answers.**")
    f = st.file_uploader("Upload a file", type=["md","txt"], key="upl_file")
    ttl = st.text_input("Optional page title", key="upl_title")
    if st.button("Upload & update knowledge", key="upl_btn"):
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
            res = incremental_add(text, str(dst).replace("\\","/"), progress=prog)
            st.success(f"✅ Incrementally indexed {res.get('added_chunks', 0)} chunks from {slug}.md")

# ===== ABOUT =====
with tab_about:
    st.markdown("### How this works")
    st.markdown("""
- **Hybrid retrieval**: semantic now; keyword layer can be added.
- **Reranker (optional)**: cross-encoder refines the top candidates.
- **Q&A**: answers strictly from retrieved passages; cites sources.
- **Context Visualizer**: each sentence maps to the strongest supporting passage.
- **Incremental add**: appends only new file vectors.
- **Targeted delete**: removes vectors for selected files (no re-embed).
- **Prebuilt index**: drop files into `data/index/prebuilt/` for instant startup.
""")
    stats_now = corpus_stats()
    st.info(f"Corpus stats: {stats_now['files']} files · ~{stats_now['size_kb']} KB")
