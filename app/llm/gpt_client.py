# app/llm/gpt_client.py
import os, json, pathlib, datetime
from typing import Tuple
from dotenv import load_dotenv

# OpenAI SDK (>=1.40)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import tiktoken
except Exception:
    tiktoken = None

load_dotenv()

PRICES = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},
}


def _enc_for(model):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text, model):
    enc = _enc_for(model)
    if enc:
        try:
            return len(enc.encode(text or ""))
        except Exception:
            pass
    # crude fallback
    return max(1, len(text or "") // 4)


class BudgetGuard:
    def __init__(self, max_daily_usd):
        self.max_daily_usd = float(max_daily_usd or 0.0)
        self.path = pathlib.Path("data/billing/usage.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def check_and_reserve(self, est):
        if self.max_daily_usd <= 0:
            return True
        today = datetime.date.today().isoformat()
        used = float(self.data.get(today, 0.0))
        if used + est > self.max_daily_usd:
            return False
        self.data[today] = used + est
        self._save()
        return True

    def adjust(self, delta):
        if self.max_daily_usd <= 0:
            return
        today = datetime.date.today().isoformat()
        self.data[today] = max(0.0, float(self.data.get(today, 0.0)) + delta)
        self._save()


class GPTClient:
    def __init__(self):
        self.use = os.getenv("USE_OPENAI", "false").lower() == "true"
        self.key = os.getenv("OPENAI_API_KEY", "") or ""
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_in = int(os.getenv("OPENAI_MAX_INPUT_TOKENS", "6000"))
        self.max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "400"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.guard = BudgetGuard(os.getenv("OPENAI_MAX_DAILY_USD", "0.50"))
        self.price = PRICES.get(self.model, PRICES["gpt-4o-mini"])

        if self.use:
            if not self.key:
                raise RuntimeError("USE_OPENAI=true but OPENAI_API_KEY is missing.")
            if OpenAI is None:
                raise RuntimeError(
                    "OpenAI SDK not available. Please ensure 'openai>=1.40.0' is installed."
                )
            self.client = OpenAI(api_key=self.key)
        else:
            self.client = None

    def _estimate_cost(self, in_t, out_t):
        return (in_t / 1e6) * self.price["in"] + (out_t / 1e6) * self.price["out"]

    def answer(self, context: str, question: str) -> Tuple[str, dict]:
        if not self.use or not self.client:
            raise RuntimeError("GPT disabled or client unavailable.")
        ctx = (context or "")[: self.max_in * 4]
        in_t = _count_tokens(ctx, self.model) + _count_tokens(question, self.model)
        est = self._estimate_cost(in_t, self.max_out)
        if not self.guard.check_and_reserve(est):
            raise RuntimeError("Daily OpenAI budget exceeded.")

        try:
            msgs = [
                {
                    "role": "system",
                    "content": "Answer concisely using ONLY the provided context. If something is not in the context, explicitly say 'Not in context.'",
                },
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{question}"},
            ]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_out,
            )
            text = (resp.choices[0].message.content or "").strip()
            out_t = _count_tokens(text, self.model)
            self.guard.adjust(self._estimate_cost(in_t, out_t) - est)
            if not text:
                # never return empty
                text = "Not in context."
            return text, {"in_tokens": in_t, "out_tokens": out_t, "llm": self.model}
        except Exception as e:
            # release provisional hold
            self.guard.adjust(-est)
            raise RuntimeError(f"OpenAI call failed: {e}")
