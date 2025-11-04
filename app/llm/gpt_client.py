# app/llm/gpt_client.py
import os, json, pathlib, datetime
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

# OpenAI SDK v1.x
from openai import OpenAI
try:
    import tiktoken
except Exception:
    tiktoken = None

# ---- Pricing per 1M tokens (USD) — adjust as needed if OpenAI updates pricing
PRICES = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},   # $/1M tokens
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},
}

def _enc_for(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str, model: str) -> int:
    enc = _enc_for(model)
    if enc:
        try:
            return len(enc.encode(text or ""))
        except Exception:
            pass
    # Fallback heuristic ~4 chars per token
    return max(1, len(text or "") // 4)

class BudgetGuard:
    """Simple daily budget guard stored in data/billing/usage.json."""
    def __init__(self, max_daily_usd: float):
        self.max_daily_usd = float(max_daily_usd or 0.0)
        self.path = pathlib.Path("data/billing/usage.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def check_and_reserve(self, est_cost_usd: float) -> bool:
        """Return True if within budget; otherwise False."""
        today = datetime.date.today().isoformat()
        used = float(self.data.get(today, 0.0))
        if used + est_cost_usd > self.max_daily_usd:
            return False
        # Reserve (optimistic); caller should write back actuals later
        self.data[today] = used + est_cost_usd
        self._save()
        return True

    def adjust(self, delta_usd: float):
        """Adjust today's usage up/down after actuals known."""
        today = datetime.date.today().isoformat()
        used = float(self.data.get(today, 0.0))
        self.data[today] = max(0.0, used + delta_usd)
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

        if self.use and not self.key:
            raise RuntimeError("USE_OPENAI=True but OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=self.key) if self.use else None
        self.price = PRICES.get(self.model, PRICES["gpt-4o-mini"])

    def _estimate_cost_usd(self, in_tokens: int, out_tokens: int) -> float:
        cin = (in_tokens / 1_000_000) * self.price["in"]
        cout = (out_tokens / 1_000_000) * self.price["out"]
        return cin + cout

    def answer(self, context: str, question: str) -> Tuple[str, dict]:
        """Return (answer_text, metadata). Might raise if API fails."""
        if not self.use:
            raise RuntimeError("GPTClient is disabled (USE_OPENAI=false).")

        # Truncate context to max input tokens
        # For a rough cut, just trim by token count
        if tiktoken:
            enc = _enc_for(self.model)
            if enc:
                ids = enc.encode(context or "")
                if len(ids) > self.max_in:
                    ids = ids[: self.max_in]
                context = enc.decode(ids)
        else:
            # Heuristic: trim chars
            approx_chars = self.max_in * 4
            context = (context or "")[:approx_chars]

        in_tokens = _count_tokens(context, self.model) + _count_tokens(question, self.model)
        est_cost = self._estimate_cost_usd(in_tokens, self.max_out)

        # Budget check (optimistic reserve)
        if self.guard.max_daily_usd > 0 and not self.guard.check_and_reserve(est_cost):
            raise RuntimeError(f"Daily OpenAI budget exceeded. Set OPENAI_MAX_DAILY_USD higher or USE_OPENAI=false.")

        try:
            messages = [
                {"role": "system", "content":
                    "You are a concise assistant. Answer ONLY using the provided context. "
                    "If the answer is not contained in the context, say you cannot find it."
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_out,
            )
            text = resp.choices[0].message.content.strip()
            # Best-effort actual usage (not all SDKs expose token counts; adjust if available)
            # If token usage metadata becomes available, replace the estimate below:
            out_tokens = _count_tokens(text, self.model)
            actual_cost = self._estimate_cost_usd(in_tokens, out_tokens)
            # adjust optimistic reserve to actuals
            self.guard.adjust(actual_cost - est_cost)
            return text, {"in_tokens": in_tokens, "out_tokens": out_tokens, "usd": actual_cost}
        except Exception as e:
            # if call failed, release the reserved estimate
            self.guard.adjust(-est_cost)
            raise
