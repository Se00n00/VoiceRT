"""Google Gemini cloud leg behind the sidecar contract.

No local weights, tokenizer, or VRAM: inference lives at the Gemini API,
so this matches the sidecar selection contract and :class:`LlmModel`
selects it via ``LlmConfig(backend="gemini")``:

- ``is_sidecar = True``: :class:`LlmModel` passes ``(messages, tools)``
  straight through instead of encoding to ids first.
- ``chat`` / ``chat_stream`` speak text; streamed pieces carry
  ``token_id=-1`` upstream (callers already fall back to joined pieces).

Tool calls come back as native function calls and are serialized to the
house ``{"action": ...}`` JSON the repo parsers already consume (same
trick as the retired cloud shims): one grammar downstream, whatever
the brain. Thinking parts ride as ``<think>`` so :func:`split_thinking`
recovers them. ``warm()`` checks the key only — no network, no cost.
"""

import json
import os
import time

__all__ = ["GeminiCloud", "DEFAULT_MODEL", "resolve_api_key"]

DEFAULT_MODEL = "gemini-3.8-flash"


def _repo_root() -> str:
    try:
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    except Exception:
        return os.getcwd()


def resolve_api_key(explicit: str = "auto") -> str:
    """Explicit key, else ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` env/.env."""
    if explicit and str(explicit) != "auto":
        return str(explicit)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=os.path.join(_repo_root(), ".env"))
        except Exception:
            pass
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ABORT: no Gemini API key. Set GEMINI_API_KEY in the "
            "environment or the gitignored .env at the repo root "
            "(free key: https://aistudio.google.com/apikey). "
            "Refusing an unauthenticated warm on purpose."
        )
    return key


def _to_contents(messages):
    """[{role, content}] -> (system_text, contents). Pure."""
    system, contents = [], []
    for m in messages or []:
        role = str(m.get("role", "user") if isinstance(m, dict)
                   else "user").lower()
        text = str(m.get("content", "") if isinstance(m, dict) else m)
        if role == "system":
            if text.strip():
                system.append(text)
        else:
            contents.append({
                "role": "model" if role in ("assistant", "ai") else "user",
                "parts": [{"text": text}],
            })
    return "\n\n".join(system), contents


def _to_declarations(tools):
    """OpenAI-shaped tool specs -> Gemini function declarations. Pure."""
    from google.genai import types

    decls = []
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = str(fn.get("name", "") or "").strip()
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object"}
        if not isinstance(params, dict):
            params = {"type": "object"}
        try:
            decls.append(types.FunctionDeclaration(
                name=name,
                description=str(fn.get("description", ""))[:500],
                parameters=params,
            ))
        except Exception:
            continue
    return decls


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


def _thought_of(part) -> str:
    if bool(getattr(part, "thought", False)):
        return str(getattr(part, "text", "") or "")
    return ""


class GeminiCloud:
    """Gemini leg. ``warm()`` validates the key; ``close()`` is a no-op."""

    is_sidecar = True

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = "auto",
                 temperature: float = 0.0):
        self.model = str(model or DEFAULT_MODEL)
        self.api_key = str(api_key or "auto")
        self.temperature = float(temperature)
        self.base_url = "https://generativelanguage.googleapis.com"
        self._client_inst = None

    # -- lifecycle ------------------------------------------------------
    def _client(self):
        if self._client_inst is None:
            from google import genai

            self._client_inst = genai.Client(api_key=resolve_api_key(self.api_key))
        return self._client_inst

    def _health(self) -> bool:
        """Key present (only). Never raises; the first call is the probe."""
        try:
            resolve_api_key(self.api_key)
            return True
        except Exception:
            return False

    def warm(self):
        """Resolve the key + construct the client. Raises on failure."""
        resolve_api_key(self.api_key)
        self._client()
        return self

    # -- chat contract ---------------------------------------------------
    def _config(self, tools, max_tokens, stop):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            max_output_tokens=int(max_tokens),
            temperature=self.temperature,  # greedy, house rule
        )
        decls = _to_declarations(tools)
        if decls:
            cfg.tools = [types.Tool(function_declarations=decls)]
        if stop:
            cfg.stop_sequences = list(stop)
        return cfg

    def _split(self, resp):
        """Response -> (thought, text, [(name, args)]). Never raises."""
        thought, texts, calls = "", [], []
        try:
            cands = getattr(resp, "candidates", None) or []
            parts = getattr(getattr(cands[0], "content", None), "parts",
                            None) if cands else None
            for p in parts or []:
                th = _thought_of(p)
                if th:
                    thought += th
                elif getattr(p, "text", None):
                    texts.append(str(p.text))
        except Exception:
            pass
        try:
            for fc in getattr(resp, "function_calls", None) or []:
                if getattr(fc, "name", ""):
                    calls.append((str(fc.name), getattr(fc, "args", {}) or {}))
        except Exception:
            pass
        if not texts:
            try:
                if getattr(resp, "text", ""):
                    texts.append(str(resp.text))
            except Exception:
                pass
        return thought.strip(), "".join(texts), calls

    def _chat_for(self, sys_text, contents, tools, max_tokens, stop):
        """Fresh Chat per call (the caller ships full history each step,
        so nothing is retained between calls). Returns (chat, last_text).

        Uses the Chat API (``send_message``), not ``Models.generate_content``:
        the SDK routes automatic function calling through Chat and warns
        against direct use on Models.
        """
        history = list(contents[:-1]) if len(contents or []) > 1 else []
        last = contents[-1] if contents else {"parts": [{"text": ""}]}
        parts = last.get("parts", [{"text": ""}]) if isinstance(
            last, dict) else [str(last)]
        first = parts[0] if parts else {"text": ""}
        text = first.get("text", "") if isinstance(first, dict) else str(first)
        cfg = self._config(tools, max_tokens, stop)
        if sys_text.strip():
            cfg.system_instruction = sys_text
        chat = self._client().chats.create(
            model=self.model, config=cfg, history=history)
        return chat, str(text)

    def chat(self, messages, tools=None, max_tokens=320, stop=None):
        """One full turn. Returns dict with composed ``text`` (raises on
        transport/API errors only — model content never raises)."""
        sys_text, contents = _to_contents(messages)
        chat, text = self._chat_for(sys_text, contents, tools, max_tokens,
                                    stop)
        t0 = time.monotonic()
        resp = chat.send_message(text)
        dt = time.monotonic() - t0
        thought, text, calls = self._split(resp)
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

        Mirrors the bonsai tail contract: live pieces accumulate in the
        caller; the tail carries tool calls only. Stats land in ``acc``.
        """
        sys_text, contents = _to_contents(messages)
        chat, text = self._chat_for(sys_text, contents, tools, max_tokens,
                                    stop)
        t0, first_at = time.monotonic(), None
        seen: dict = {}
        stream = chat.send_message_stream(text)
        for chunk in stream:
            if first_at is None:
                first_at = time.monotonic()
            try:
                cands = getattr(chunk, "candidates", None) or []
                parts = getattr(getattr(cands[0], "content", None), "parts",
                                None) if cands else None
                for p in parts or []:
                    th = _thought_of(p)
                    if th:
                        yield ("think", th)
                    elif getattr(p, "text", None):
                        yield ("text", str(p.text))
            except Exception:
                continue
            try:
                for fc in getattr(chunk, "function_calls", None) or []:
                    if getattr(fc, "name", ""):
                        seen[str(fc.name)] = getattr(fc, "args", {}) or {}
            except Exception:
                continue
        acc["ttft"] = (first_at - t0) if first_at else 0.0
        acc["decode_tps"] = 0.0
        ordered = [(n, a) for n, a in seen.items()]
        acc["tool_calls"] = [
            {"function": {"name": n, "arguments": a}} for n, a in ordered
        ]
        if ordered:
            yield ("text", _house_json(*ordered[0]))
