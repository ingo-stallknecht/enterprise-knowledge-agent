# app/actions.py
"""
Simple file-based wiki actions used by tests and scripts.

In the current Streamlit-only setup the UI uses its own helper functions,
but these actions are still useful for a minimal programmatic API and tests.
"""

from pathlib import Path
from typing import Dict

from slugify import slugify

WIKI_ROOT = Path("data/processed/wiki")


def _ensure_wiki_root() -> Path:
    WIKI_ROOT.mkdir(parents=True, exist_ok=True)
    return WIKI_ROOT


def _slugify(title: str) -> str:
    """Slugify a page title into a safe filename (without extension)."""
    base = title or "page"
    return slugify(base, lowercase=True, separator="-")


def upsert_wiki_page(title: str, content: str) -> Dict[str, str]:
    """
    Create or overwrite a wiki page under data/processed/wiki.

    Returns metadata with title, slug, and path.
    """
    root = _ensure_wiki_root()
    slug = _slugify(title)
    path = root / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return {
        "title": title,
        "slug": slug,
        "path": str(path),
    }


def delete_wiki_page(slug: str) -> bool:
    """
    Delete a wiki page by slug (without extension).

    Returns True if the file existed and was removed, False otherwise.
    """
    root = _ensure_wiki_root()
    path = root / f"{slug}.md"
    if not path.exists():
        return False
    path.unlink()
    return True
