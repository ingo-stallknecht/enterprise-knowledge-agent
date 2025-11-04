import os, re, json, time, pathlib
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


BASE = "https://about.gitlab.com/handbook/"
RAW_DIR = pathlib.Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = pathlib.Path("data/processed"); PROC_DIR.mkdir(parents=True, exist_ok=True)


SEED_PAGES = [
BASE,
BASE + "values/",
BASE + "engineering/",
BASE + "people-operations/",
]


HEADERS = {"User-Agent": "EKA-Ingest/1.0"}


visited = set()


def fetch(url):
r = requests.get(url, headers=HEADERS, timeout=20)
r.raise_for_status()
return r.text


slugify = lambda s: re.sub(r"[^a-z0-9-]","-", s.lower())


links = []
for seed in SEED_PAGES:
try:
html = fetch(seed)
soup = BeautifulSoup(html, "html.parser")
for a in soup.select("a[href]"):
href = a.get("href")
if not href: continue
if href.startswith("/handbook/"):
links.append("https://about.gitlab.com" + href)
elif href.startswith(BASE):
links.append(href)
except Exception as e:
print("seed error", seed, e)


# unique + within handbook
links = [l.split("#")[0] for l in links if l.startswith(BASE)]
links = sorted(set(links))[:300] # cap for demo; increase for fuller crawl


print(f"Found {len(links)} pages")


for url in links:
if url in visited: continue
try:
html = fetch(url)
md_text = md(html)
slug = slugify(url.replace(BASE, "").strip("/")) or "index"
(RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
(PROC_DIR / f"{slug}.md").write_text(md_text, encoding="utf-8")
visited.add(url)
time.sleep(0.2)
except Exception as e:
print("fetch fail", url, e)


print("DONE", len(visited))