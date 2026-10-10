"""Groq cloud leg behind the sidecar contract.

No local weights, tokenizer, or VRAM: inference lives at the Groq API
(OpenAI-compatible ``/openai/v1``), so this matches the sidecar selection
contract and :class:`LlmModel` selects it via ``LlmConfig(backend="groq")``:

- ``is_sidecar = True``: :class:`LlmModel` passes ``(messages, tools)``
  straight through instead of encoding to ids first.
- ``chat`` / ``chat_stream`` speak text; streamed pieces carry
  ``token_id=-1`` upstream (callers already fall back to joined pieces).

Tool calls come back as native function calls and are serialized to the
house ``{"action": ...}`` JSON the repo parsers already consume (same
trick as the Gemini leg): one grammar downstream, whatever the brain.
Reasoning parts (``reasoning``/``reasoning_content``) ride as ``<think>`` so
:func:`split_thinking` recovers them. ``warm()`` checks the key only —
no network, no cost.
"""

import json
import os
import time

__all__ = ["GroqCloud", "DEFAULT_MODEL", "BASE_URL", "resolve_api_key"]

DEFAULT_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"


def _repo_root() -> str:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def resolve_api_key(explicit: str = "auto") -> str:
    """Explicit key, else ``GROQ_API_KEY`` env/.env."""
    if explicit and str(explicit) != "auto":
        return str(explicit)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=os.path.join(_repo_root(), ".env"))
        except Exception:
            pass
        key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ABORT: no Groq API key. Set GROQ_API_KEY in the "
            "environment or the gitignored .env at the repo root "
            "(free key: https://console.groq.com/keys). "
            "Refusing an unauthenticated warm on purpose."
        )
    return key


def _house_json(name: str, args) -> str:
    """One native call -> house ``{"action": ...}`` JSON text. Pure."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return json.dumps({"action": name, **args})


class GroqCloud:
    """Groq leg. ``warm()`` validates the key; ``close()`` is a no-op."""

    is_sidecar = True

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = "auto",
                 temperature: float = 0.0,
                 base_url: str = BASE_URL,
                 _post_fn=None, _stream_fn=None):
        self.model = str(model or DEFAULT_MODEL)
        self.api_key = str(api_key or "auto")
        self.temperature = float(temperature)
        self.base_url = str(base_url or BASE_URL)
        self._post_fn = _post_fn
        self._stream_fn = _stream_fn

    # -- lifecycle ------------------------------------------------------
    def _health(self) -> bool:
        """Key present (only). Never raises; the first call is the probe."""
        try:
            resolve_api_key(self.api_key)
            return True
        except Exception:
            return False

    def warm(self):
        """Resolve the key. Raises on failure; no network, no cost."""
        resolve_api_key(self.api_key)
        return self

    def close(self):
        return None

    # -- HTTP (injectable for tests) ------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {resolve_api_key(self.api_key)}"}

    def _post(self, path, payload, timeout):
        if self._post_fn is not None:
            return self._post_fn(path, payload, timeout)
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout) as c:
            r = c.post(path, json=payload, headers=self._headers())
            r.raise_for_status()
            return r.json()

    def _stream(self, path, payload, timeout):
        if self._stream_fn is not None:
            yield from self._stream_fn(path, payload, timeout)
            return
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout) as c:
            with c.stream("POST", path, json=payload,
                          headers=self._headers()) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    yield line

    # -- chat contract ---------------------------------------------------
    def _payload(self, messages, tools, max_tokens, stop, stream):
        from src.models.llm import to_openai_tools

        p = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": self.temperature,  # greedy, house rule
            "stream": bool(stream),
        }
        oai = to_openai_tools(tools)
        if oai:
            p["tools"] = oai
        if stop:
            p["stop"] = list(stop)
        return p

    @staticmethod
    def _split_msg(msg):
        """One OpenAI message dict -> (thought, text, [(name, args)]). Pure.

        Reasoning location varies by model family: gpt-oss uses
        ``reasoning`` (chat) / ``delta.reasoning`` (stream), others
        (deepseek, qwen) use ``reasoning_content``. Read both.
        """
        thought = str(msg.get("reasoning_content", "")
                      or msg.get("reasoning", "") or "").strip()
        text = str(msg.get("content", "") or "")
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name", "") or "")
            if name:
                calls.append((name, fn.get("arguments", "") or ""))
        return thought, text, calls

    def chat(self, messages, tools=None, max_tokens=320, stop=None):
        """One full turn. Returns dict with composed ``text`` (raises on
        transport/API errors only — model content never raises)."""
        t0 = time.monotonic()
        out = self._post("/chat/completions",
                         self._payload(messages, tools, max_tokens, stop,
                                       False),
                         timeout=120)
        dt = time.monotonic() - t0
        msg = (out.get("choices") or [{}])[0].get("message", {})
        thought, text, calls = self._split_msg(msg)
        if calls:
            # Structured call wins: house JSON is what the parsers eat.
            text = _house_json(*calls[0])
        if thought:
            text = f"<think>{thought}</think>{text}"
        return {
            "text": text,
            "tool_calls": [
                {"function": {"name": n, "arguments": a}} for n, a in calls
            ],
            "ttft": dt,  # blocking call: whole round trip, honest label
            "decode_tps": 0.0,  # no server timings on this API
        }

    def chat_stream(self, acc, messages, tools=None, max_tokens=320,
                    stop=None):
        """Yield ("text"|"think", piece) live, then one composed tool block.

        Mirrors the tail contract: live pieces accumulate in the caller;
        the tail carries tool calls only. Stats land in ``acc``.
        """
        t0, first_at = time.monotonic(), None
        tcs: dict = {}
        for line in self._stream(
                "/chat/completions",
                self._payload(messages, tools, max_tokens, stop, True),
                timeout=300):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if first_at is None:
                first_at = time.monotonic()
            delta = (ev.get("choices") or [{}])[0].get("delta", {})
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                yield ("think", str(think))
            if delta.get("content"):
                yield ("text", str(delta["content"]))
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tcs.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = str(fn["name"])
                if fn.get("arguments"):
                    slot["arguments"] += str(fn["arguments"])
        acc["ttft"] = (first_at - t0) if first_at else 0.0
        acc["decode_tps"] = 0.0
        ordered = [(v["name"], v["arguments"]) for _, v in sorted(tcs.items())
                   if v["name"]]
        acc["tool_calls"] = [
            {"function": {"name": n, "arguments": a}} for n, a in ordered
        ]
        if ordered:
            yield ("text", _house_json(*ordered[0]))
