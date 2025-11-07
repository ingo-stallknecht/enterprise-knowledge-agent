# scripts/fetch_gitlab_handbook.py
"""
Fetch a curated subset of the GitLab Handbook (public pages).
Stores raw HTML and markdownified text under data/raw and data/processed.
Skips gated pages (e.g., culture/ → sign-in).
"""

import time, pathlib, requests
from markdownify import markdownify as md

RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
PROC = pathlib.Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "EKA-Ingest/1.0"}
CURATED = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/handbook/values/",
    "https://about.gitlab.com/handbook/engineering/",
    "https://about.gitlab.com/handbook/people-group/",
    "https://about.gitlab.com/handbook/communication/",
    "https://about.gitlab.com/handbook/product/",
    "https://about.gitlab.com/handbook/sales/",
    "https://about.gitlab.com/handbook/marketing/",
    # high-signal pages; avoid gated culture/ URL
    "https://about.gitlab.com/handbook/leadership/",
    "https://about.gitlab.com/handbook/engineering/management/",
]

def slug(url: str) -> str:
    s = url.split("https://about.gitlab.com/handbook/")[-1].strip("/")
    return (s or "index").replace("/", "-")

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    r.raise_for_status()
    # crude gate detection: some protected pages redirect to gitlab.com sign_in
    if "gitlab.com/users/sign_in" in r.url:
        raise requests.HTTPError(f"Gated page redirected to sign-in: {r.url}")
    return r.text

def main():
    ok = 0
    for i, url in enumerate(CURATED, 1):
        try:
            html = fetch(url)
            (RAW / f"{slug(url)}.html").write_text(html, encoding="utf-8")
            (PROC / f"{slug(url)}.md").write_text(md(html), encoding="utf-8")
            print(f"[{i}/{len(CURATED)}] ok: {url}")
            ok += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[{i}/{len(CURATED)}] skip: {url} -> {e}")
    print(f"[done] downloaded {ok} pages")

if __name__ == "__main__":
    main()
