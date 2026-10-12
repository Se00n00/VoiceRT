"""Token budget for L1 assembly: estimate + per-backend rows.

One estimator, two live consumers:
- the L1 window (``LangChainSessionMemory`` token prune via
  ``langchain_core.messages.utils.trim_messages``),
- the summarization trigger (``token_counter=count_messages``).

Estimation order: llama-server ``/tokenize`` when a sidecar is warm
(exact, one local HTTP call, cached per message) → ``chars/3.5`` fallback
(mixed tool text runs denser than prose; 4.0 undercounts JSON/URLs).

Budget shape (16K reference — see README "Context Engineering")::

    16K window
    ├── system + preamble ......... fixed (~2.5K, never cut)
    ├── tool specs ................ fixed (~1K narrowed, never cut)
    ├── history window ............ elastic (newest-first fill to remainder)
    ├── observations .............. elastic per-call (1500 → 500 → drop)
    └── generation reserve ........ fixed floor (256/512, never spent by input)

Coherence rule: step floor >= think cap + min answer. A think cap above
the floor truncates mid-thought — decorative, not enforced.
"""
import json
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "estimate_tokens",
    "count_messages",
    "tokenize_count",
    "CONTEXT_TOKENS",
    "BudgetConfig",
    "BUDGETS",
    "for_backend",
]

# Native context per backend. Bonsai ships 262K but runs at 16K here
# (measured: prefill ~100 tok/s to 13.5K, decode ~1.1 tok/s at ngl=6).
CONTEXT_TOKENS = {
    "bonsai": 16384,
    "qwen": 8192,
    "groq": 131072,
    "gemini": 65536,
}
# Think + tool call must fit in ONE step (matches chat_model floors).
# Heuristic for mixed chat/tool text when no tokenizer is reachable.
CHARS_PER_TOKEN = 3.5


class BudgetConfig(BaseModel):
    """Per-backend context budget. Fixed parts are caps (never cut);
    elastic parts fill whatever remains after the fixed parts."""

    model_config = ConfigDict(frozen=True)

    ctx: int
    system_cap: int = 2500
    specs_cap: int = 1000
    # observation shrink stages in chars: full→first→second→drop(0)
    obs_steps: tuple = (1500, 500, 0)
    # max reasoning tokens; None = unbounded (server default)
    think_cap: int | None = 128
    # generation reserve: never spent by input (see coherence rule)
    step_floor: int = 512
    headroom: float = 0.8


BUDGETS = {
    # 16K reference row: matches the README diagram exactly.
    "bonsai": BudgetConfig(ctx=16384, think_cap=128, step_floor=512),
    "qwen": BudgetConfig(ctx=8192, think_cap=256, step_floor=256),
}

DEFAULT_BUDGET = BudgetConfig(ctx=4096)


def for_backend(name: str) -> BudgetConfig:
    """Budget row by backend id (prefix match: qwen06 -> qwen)."""
    name = str(name or "")
    if name in BUDGETS:
        return BUDGETS[name]
    for key in ("qwen", "bonsai"):
        if name.startswith(key):
            return BUDGETS[key]
    return DEFAULT_BUDGET


def count_messages(messages) -> int:
    """Token-counter adapter for middleware (heuristic, no server calls).

    Accepts LangChain messages, (role, content) tuples, dicts or plain
    strings; tool-call args count too (they consume context as JSON).
    Single estimate call over the joined text.
    """
    parts: list[str] = []
    for m in messages or []:
        try:
            if isinstance(m, str):
                parts.append(m)
            elif isinstance(m, dict):
                parts.append(str(m.get("content", "") or ""))
                for tc in m.get("tool_calls", None) or []:
                    parts.append(json.dumps(tc, default=str))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                parts.append(str(m[1]))
            else:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    bits = []
                    for b in content:
                        if isinstance(b, dict):
                            bits.append(str(b.get("text", b)))
                        else:
                            bits.append(str(b))
                    parts.append(" ".join(bits))
                elif content:
                    parts.append(str(content))
                for tc in getattr(m, "tool_calls", None) or []:
                    parts.append(json.dumps(
                        tc if isinstance(tc, dict) else getattr(
                            tc, "args", tc), default=str))
        except Exception:
            continue
    return estimate_tokens("\n".join(parts))


_tok_cache: dict = {}


def tokenize_count(base_url: str, text: str, timeout: float = 10.0):
    """Exact count via llama-server /tokenize. None on any failure."""
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/tokenize",
            data=json.dumps({"content": str(text or "")}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        toks = data.get("tokens", [])
        return len(toks) if isinstance(toks, list) else None
    except Exception:
        return None


def estimate_tokens(text: str, base_url: str | None = None) -> int:
    """Best-effort token count: /tokenize (cached) else chars/3.5."""
    s = str(text or "")
    if not s:
        return 0
    if base_url:
        key = (base_url, s)
        if key not in _tok_cache:
            n = tokenize_count(base_url, s)
            _tok_cache[key] = n if n is not None else -1
            if len(_tok_cache) > 4096:
                _tok_cache.clear()
        hit = _tok_cache[key]
        if hit >= 0:
            return hit
    return max(1, int(len(s) / CHARS_PER_TOKEN))
