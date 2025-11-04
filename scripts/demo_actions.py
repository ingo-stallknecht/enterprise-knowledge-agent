import json
from app.agent import KnowledgeAgent
from app.actions import upsert_wiki_page, create_ticket


agent = KnowledgeAgent()


print("Retrieve demo:")
print(json.dumps(agent.retrieve("company values"), indent=2))


print("\nAnswer with citations:")
ans = agent.answer_with_citations("How to propose a change to the handbook?")
print(json.dumps(ans, indent=2))


print("\nUpsert wiki page:")
print(upsert_wiki_page("Product Onboarding Checklist", "- Step 1...\n- Step 2..."))


print("\nCreate ticket:")
print(create_ticket("Broken page link", "The link to values page 404s.", "high"))