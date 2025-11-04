# scripts/test_api_local.py
import requests, json, time

BASE = "http://127.0.0.1:8000"

def post(path, payload):
    r = requests.post(f"{BASE}{path}", json=payload)
    print(f"\n[{path}] status={r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2)[:1000])  # preview only
    except Exception:
        print(r.text[:1000])

if __name__ == "__main__":
    # Wait a bit if server just started
    time.sleep(1)

    # 1️⃣ Health check
    r = requests.get(f"{BASE}/healthz")
    print("[/healthz]", r.status_code, r.json())

    # 2️⃣ Retrieval
    post("/retrieve", {"query": "company values", "k": 3})

    # 3️⃣ Answer with citations
    post("/answer_with_citations", {
        "question": "How can I propose a change to the handbook?",
        "k": 4
    })

    # 4️⃣ Wiki upsert
    post("/upsert_wiki_page", {
        "title": "Demo Page",
        "content": "# Test page\nThis page was created by test_api_local."
    })

    # 5️⃣ Ticket creation
    post("/create_ticket", {
        "title": "Broken link in engineering page",
        "body": "Link to CI/CD guide returns 404",
        "priority": "high"
    })

