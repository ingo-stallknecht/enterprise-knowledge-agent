# scripts/fetch_gitlab_handbook.py
import os
import re
import subprocess
import sys
import pathlib
import time
from typing import List, Optional

import requests
from markdownify import markdownify as md
from bs4 import BeautifulSoup

RAW_DIR = pathlib.Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = pathlib.Path("data/processed"); PROC_DIR.mkdir(parents=True, exist_ok=True)
REPO_DIR = pathlib.Path("data/repo")  # where we clone the website repo
REPO_URL = "https://gitlab.com/gitlab-com/www-gitlab-com.git"

# Fallback curated URLs (in case git clone isn't possible)
CURATED_URLS = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/handbook/values/",
    "https://about.gitlab.com/handbook/engineering/",
    "https://about.gitlab.com/handbook/people-group/",
    "https://about.gitlab.com/handbook/communication/",
    "https://about.gitlab.com/handbook/product/",
    "https://about.gitlab.com/handbook/sales/",
    "https://about.gitlab.com/handbook/marketing/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://about.gitlab.com/",
}

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "-", s.lower()).strip("-")

def have_git() -> bool:
    try:
        subprocess.check_output(["git", "--version"])
        return True
    except Exception:
        return False

def run(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def strip_front_matter(text: str) -> str:
    """
    Remove Jekyll/Hugo-style front matter delimited by '---' at the top of the file.
    Falls back gracefully if none present.
    """
    if text.startswith("---"):
        # match until the next --- on its own line
        m = re.search(r"^---\s*$.*?^---\s*$\n?", text, flags=re.DOTALL | re.MULTILINE)
        if m:
            return text[m.end():]
    return text

def ingest_from_repo() -> int:
    """
    Sparse, shallow clone of only the handbook content to avoid huge size
    and Windows-illegal paths elsewhere.
    """
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    repo_path = REPO_DIR / "www-gitlab-com"

    # if previous failed checkout exists, nuke it (Windows-safe)
    if repo_path.exists():
        import shutil
        shutil.rmtree(repo_path, ignore_errors=True)

    # sparse, shallow, blob-less
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(repo_path)])
    # only check out the handbook subtree
    run(["git", "sparse-checkout", "set", "source/handbook"], cwd=repo_path)

    # Collect *.md under the handbook subtree
    md_files: List[pathlib.Path] = []
    for p in (repo_path / "source" / "handbook").rglob("*"):
        if p.suffix.lower() in {".md", ".mdx"} and p.is_file():
            md_files.append(p)

    md_files = sorted(set(md_files))
    print(f"[repo] sparse handbook files: {len(md_files)}")

    count = 0
    for i, fp in enumerate(md_files, start=1):
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            text = strip_front_matter(raw)

            rel = fp.relative_to(repo_path / "source" / "handbook")
            slug_part = str(rel).replace("\\", "/")
            slug_part = re.sub(r"/index\\.mdx?$", "", slug_part)
            slug = slugify(slug_part) or "index"

            (RAW_DIR / f"{slug}.md").write_text(raw, encoding="utf-8")
            (PROC_DIR / f"{slug}.md").write_text(text, encoding="utf-8")
            count += 1
            if i % 50 == 0:
                print(f"[repo] processed {i}/{len(md_files)}")
        except Exception as e:
            print("[repo] skip", fp, "→", e)
    print(f"[repo] DONE, processed {count} pages")
    return count

def fetch_ok(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
    if r.url.startswith("https://gitlab.com/users/sign_in"):
        raise requests.HTTPError("Redirected to gitlab sign-in (skipping).")
    r.raise_for_status()
    return r

def ingest_fallback_http(urls: List[str], delay_s: float = 0.25) -> int:
    """
    Fallback: fetch a curated set of public handbook pages via HTTP and convert to markdown.
    """
    print("[fallback] using curated URL list")
    done = 0
    for i, url in enumerate(urls, start=1):
        try:
            r = fetch_ok(url)
            html = r.text
            md_text = md(html)
            # build a slug from URL path after '/handbook/'
            m = re.search(r"/handbook/(.*)$", url)
            slug_part = (m.group(1).strip("/") if m else "index")
            slug = slugify(slug_part) or "index"
            (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
            (PROC_DIR / f"{slug}.md").write_text(md_text, encoding="utf-8")
            done += 1
            print(f"[{i}/{len(urls)}] ok: {url}")
            time.sleep(delay_s)
        except Exception as e:
            print(f"[{i}/{len(urls)}] skip: {url} → {e}")
    print(f"[fallback] DONE, downloaded {done} pages")
    return done

if __name__ == "__main__":
    total = 0
    if have_git():
        try:
            total = ingest_from_repo()
        except Exception as e:
            print("[repo] failed, falling back to HTTP:", e)
    else:
        print("[repo] git not available, falling back to HTTP")

    if total == 0:
        total = ingest_fallback_http(CURATED_URLS)

    print(f"[main] FINISHED. pages={total}")
