# tests/test_actions.py
import pathlib

from app.actions import upsert_wiki_page, delete_wiki_page


def test_upsert_creates_file(tmp_path, monkeypatch):
    # Work in a temporary directory so we don't touch real repo data
    monkeypatch.chdir(tmp_path)

    out = upsert_wiki_page("QA Policy", "Always add citations.")
    path = pathlib.Path(out["path"])

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Always add citations." in text


def test_delete_wiki_page(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    out = upsert_wiki_page("Delete Me", "Temporary content")
    slug = out["slug"]

    assert delete_wiki_page(slug) is True
    # Deleting again should return False but not raise
    assert delete_wiki_page(slug) is False
