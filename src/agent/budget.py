"""Token budget for L1 assembly: estimate, window, pack.

One estimator, two consumers:
- the L1 window (how much history fits — replaces the flat 8-turn cap),
- pack() (how much observation fits — the backstop over trim + caps).

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
import urllib.request
from dataclasses import dataclass, field

__all__ = [
    "estimate_tokens",
    "count_messages",
    "tokenize_count",
    "window_slice",
    "pack_prompt",
    "CONTEXT_TOKENS",
    "STEP_FLOORS",
    "BudgetConfig",
    "BUDGETS",
    "for_backend",
    "history_budget",
]

# Native context per backend. Bonsai ships 262K but runs at 16K here
# (measured: prefill ~100 tok/s to 13.5K, decode ~1.1 tok/s at ngl=6).
CONTEXT_TOKENS = {
    "bonsai": 16384,
    "qwen": 8192,
}
# Think + tool call must fit in ONE step (matches chat_model floors).
STEP_FLOORS = {
    "bonsai": 512,
}
DEFAULT_STEP_FLOOR = 256
# Fraction of ctx reserved for generation + safety margin.
HEADROOM = 0.8
# Heuristic for mixed chat/tool text when no tokenizer is reachable.
CHARS_PER_TOKEN = 3.5


@dataclass(frozen=True)
class BudgetConfig:
    """Per-backend context budget. Fixed parts are caps (never cut);
    elastic parts fill whatever remains after the fixed parts."""

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


def history_budget(cfg: BudgetConfig | None = None,
                   backend: str = "") -> int:
    """Tokens left for history after fixed parts + reserve."""
    cfg = cfg or (for_backend(backend) if backend else DEFAULT_BUDGET)
    room = (int(cfg.ctx * cfg.headroom) - cfg.step_floor
            - cfg.system_cap - cfg.specs_cap)
    return max(0, room)


def count_messages(messages) -> int:
    """Token-counter adapter for middleware (heuristic, no server calls).

    Accepts LangChain messages, (role, content) tuples, dicts or plain
    strings; tool-call args count too (they consume context as JSON).
    Single estimate call over the joined text.
    """
    import json as _json

    parts: list[str] = []
    for m in messages or []:
        try:
            if isinstance(m, str):
                parts.append(m)
            elif isinstance(m, dict):
                parts.append(str(m.get("content", "") or ""))
                for tc in m.get("tool_calls", None) or []:
                    parts.append(_json.dumps(tc, default=str))
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
                    parts.append(_json.dumps(
                        tc if isinstance(tc, dict) else getattr(
                            tc, "args", tc), default=str))
        except Exception:
            continue
    return estimate_tokens("\n".join(parts))


_tok_cache: dict = {}


def tokenize_count(base_url: str, text: str, timeout: float = 10.0):
    """Exact count via llama-server /tokenize. None on any failure."""
    import json

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


def window_slice(messages: list, budget: int, base_url: str | None = None,
                 keep_last: int = 1) -> list:
    """Newest-first fill: keep messages while cumsum(tokens) <= budget.

    Always keeps at least ``keep_last`` trailing messages (default 1 —
    the current turn is never dropped). Pure; input order preserved.
    """
    if not messages:
        return []

    def _text(m):
        if isinstance(m, dict):
            return str(m.get("content", "") or "")
        return str(m or "")
    counts = [estimate_tokens(_text(m), base_url) for m in messages]
    total, start = 0, len(messages)
    for i in range(len(messages) - 1, -1, -1):
        total += counts[i]
        if total > budget and (len(messages) - i) > keep_last:
            start = i + 1
            break
        start = i
    return messages[start:]


def pack_prompt(parts: dict, ctx: int, step_floor: int = DEFAULT_STEP_FLOOR,
                headroom: float = HEADROOM,
                base_url: str | None = None,
                budget: BudgetConfig | None = None) -> dict:
    """Fit an assembled prompt into headroom*ctx. Cut order (never system):

    1. shrink observations through the obs stages (default 1500→500→drop),
    2. drop oldest history messages first (newest-first fill),
    3. report what was cut (cut_obs, cut_hist counts).

    ``parts``: {"system": str, "tools": str, "history": [str...],
    "trailer": str, "observation": str}. Returns the same shape with
    "history" possibly shortened, "observation" possibly shrunk, plus
    "prompt_tokens" (estimate) and "cut": {"obs": bool, "hist": int}.
    Pass ``budget`` for the per-backend obs stages (else the defaults).
    """
    stages = budget.obs_steps if budget is not None else (1500, 500, 0)
    limit = int(ctx * headroom) - int(step_floor)
    system = str(parts.get("system", "") or "")
    tools = str(parts.get("tools", "") or "")
    trailer = str(parts.get("trailer", "") or "")
    obs = str(parts.get("observation", "") or "")
    hist = [str(h or "") for h in (parts.get("history") or [])]

    def cost(h, o):
        n = estimate_tokens(system, base_url)
        n += estimate_tokens(tools, base_url)
        n += estimate_tokens(trailer, base_url)
        n += estimate_tokens(o, base_url)
        for m in h:
            n += estimate_tokens(m, base_url)
        return n

    cut_obs, cut_hist = False, 0
    for stage in stages[1:]:
        if cost(hist, obs) <= limit or not obs:
            break
        obs = obs[:stage] if stage > 0 else ""
        cut_obs = True
    # first stage caps an oversized observation even under budget
    if obs and len(obs) > stages[0]:
        obs = obs[:stages[0]]
        cut_obs = True
    if cost(hist, obs) > limit and hist:
        before = len(hist)
        # newest-first fill over history within remaining budget
        fixed = (estimate_tokens(system, base_url)
                 + estimate_tokens(tools, base_url)
                 + estimate_tokens(trailer, base_url)
                 + estimate_tokens(obs, base_url))
        hist = window_slice(hist, max(0, limit - fixed), base_url)
        cut_hist = before - len(hist)
    return {"system": system, "tools": tools, "history": hist,
            "trailer": trailer, "observation": obs,
            "prompt_tokens": cost(hist, obs),
            "cut": {"obs": cut_obs, "hist": cut_hist}}
