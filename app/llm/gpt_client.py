# app/llm/gpt_client.py
import os, json, pathlib, datetime
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

try:
    import tiktoken
except Exception:
    tiktoken = None

load_dotenv()

# $ per 1M tokens (adjust if pricing changes)
PRICES = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},
}

def _enc_for(model):
    if not tiktoken: return None
    try: return tiktoken.encoding_for_model(model)
    except Exception: return tiktoken.get_encoding("cl100k_base")

def _count_tokens(text, model):
    enc = _enc_for(model)
    if enc:
        try: return len(enc.encode(text or ""))
        except Exception: pass
    return max(1, len(text or "") // 4)

class BudgetGuard:
    """Simple daily budget guard stored in data/billing/usage.json."""
    def __init__(self, max_daily_usd):
        self.max_daily_usd = float(max_daily_usd or 0.0)
        self.path = pathlib.Path("data/billing/usage.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
    def _load(self):
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
    def _save(self): self.path.write_text(json.dumps(self.data, indent=2))
    def check_and_reserve(self, est):
        today = datetime.date.today().isoformat()
        used = float(self.data.get(today, 0.0))
        if used + est > self.max_daily_usd: return False
        self.data[today] = used + est; self._save(); return True
    def adjust(self, delta):
        today = datetime.date.today().isoformat()
        self.data[today] = max(0.0, float(self.data.get(today, 0.0)) + delta)
        self._save()

class GPTClient:
    def __init__(self):
        self.use = os.getenv("USE_OPENAI", "false").lower() == "true"
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_in = int(os.getenv("OPENAI_MAX_INPUT_TOKENS", "6000"))
        self.max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "400"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.guard = BudgetGuard(os.getenv("OPENAI_MAX_DAILY_USD", "0.50"))
        self.client = OpenAI(api_key=self.key) if self.use else None
        self.price = PRICES.get(self.model, PRICES["gpt-4o-mini"])

    def _estimate_cost(self, in_t, out_t):
        return (in_t/1e6)*self.price["in"] + (out_t/1e6)*self.price["out"]

    def answer(self, context: str, question: str) -> Tuple[str, dict]:
        if not self.use:
            raise RuntimeError("GPT disabled")
        # Trim context to input budget (approx chars)
        ctx = (context or "")[: self.max_in * 4]
        in_t = _count_tokens(ctx, self.model) + _count_tokens(question, self.model)
        est = self._estimate_cost(in_t, self.max_out)
        if self.guard.max_daily_usd > 0 and not self.guard.check_and_reserve(est):
            raise RuntimeError("Daily OpenAI budget exceeded.")
        try:
            msgs = [
                {"role": "system", "content": "Answer concisely using ONLY the provided context. If not in context, say so."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{question}"}
            ]
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=self.temperature,
                max_tokens=self.max_out
            )
            text = resp.choices[0].message.content.strip()
            out_t = _count_tokens(text, self.model)
            self.guard.adjust(self._estimate_cost(in_t, out_t) - est)
            return text, {"in_tokens": in_t, "out_tokens": out_t}
        except Exception:
            self.guard.adjust(-est)
            raise

