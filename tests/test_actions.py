# tests/test_actions.py
from app.actions import upsert_wiki_page, create_ticket


def test_upsert_and_ticket():
out = upsert_wiki_page("QA Policy", "Always add citations.")
assert out["status"] == "ok"
out2 = create_ticket("Fix typo", "Typo in onboarding page", "low")
assert out2["status"] == "ok"