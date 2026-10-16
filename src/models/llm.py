"""LLM leg: Qwen chat generation behind a clean async class."""
import asyncio
import json
import re

from pydantic import BaseModel, ConfigDict, Field

from src.prompts.base import SYSTEM_PROMPT  # noqa: F401  (re-exported for back-compat)

# Sidecar truth lives on LlmModel.SIDECAR_BACKENDS below; cloud backends
# are stock LangChain chat models via src/models/provider.py.

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_HERMES_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE)


class LlmConfig(BaseModel):
    """No YAML: construct (or override fields) in code."""

    model_config = ConfigDict(frozen=True)

    model: str = "Qwen/Qwen3-0.6B"
    # Backend switch.
    #  - "qwen" = fused Qwen3 path — DEFAULT: 0.6B fused on CUDA, ~1.4GB
    #    VRAM, no extra download on a box that already has the weights.
    #    NOTE: a 0.6B is a chat/routing brain, not a tool
    #    harness — the agentic leg wants "bonsai" below.
    #  - "bonsai" = Ternary Bonsai 2 27B via Prism-fork llama.cpp sidecar
    #    (-ngl auto from free VRAM, 16K ctx) — the WORKER brain for the
    #    full tool harness; STT/TTS keep the GPU, the rest offloads.
    #  - cloud backends = stock LangChain chat models (no local
    #    weights) — use src/models/provider.ChatModel.get, not here.
    backend: str = "qwen"
    max_tokens: int = 48
    max_seq: int = 8192
    device: str = "cuda"
    system_prompt: str = SYSTEM_PROMPT
    thinking: bool = False  # Qwen3 <think> traces (stripped from output)
    # Ternary Bonsai 2 27B via Prism-fork llama.cpp sidecar (OFFLOAD:
    # -ngl auto from free VRAM — partial on 4GB, full on T4+).
    # Select with backend="bonsai". Needs prism-b10658+ server binary.
    bonsai_gguf: str = "auto"  # explicit GGUF path or "auto" (HF cache)
    bonsai_packing: str = "ptq1_0"  # or "pq2_0" (bigger, faster prefill)
    bonsai_mmproj: str = "auto"  # vision tower, or "skip" for text-only
    bonsai_port: int = 8081
    bonsai_ctx: int = 16384
    bonsai_ngl: str = "auto"  # int, or "auto" = probe free VRAM at warm
    bonsai_cache_k: str = "q4_0"
    bonsai_threads: int = 0  # 0 = server default
    bonsai_thinking: int | None = 128  # reasoning budget, None = server
    # default (unbounded think — minutes per step at ~1 tok/s on 4GB),
    # 0 disables it. 128 is the low leash: short traces stay live in the
    # think bubble, runaway reasoning gets cut instead of stalling the
    # turn. Agentic scores were measured thinking-on; raise or set None
    # when quality matters more than latency.
    bonsai_bin: str = "auto"  # explicit Prism-fork llama-server or "auto"
    # Cloud model names (used by src/models/provider, kept here so old
    # configs with cloud backends still parse; LlmModel itself
    # no longer serves cloud backends).
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str = "auto"
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = "auto"
    groq_reasoning_effort: str = "low"
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_api_key: str = "auto"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = "auto"
    openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_api_key: str = "auto"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str = "auto"
    compat_model: str = ""
    compat_api_key: str = "auto"
    compat_base_url: str = ""


class LlmResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = ""
    ttft_s: float = 0.0
    tps: float = 0.0
    output_ids: tuple = Field(default_factory=tuple)
    # Template-native tool calls ((name, args), ...) in emit order.
    # The chat template renders tools in its own grammar, so only it
    # knows how to read them back — extraction lives here, beside
    # encode(), never in the LangChain face above this leg.
    tool_calls: tuple = Field(default_factory=tuple)


class LlmToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_id: int = 0
    piece: str = ""
    first: bool = False
    # "text" or "think": sidecar legs tag reasoning deltas so callers can
    # render thinking live without polluting the answer preview. Default
    # keeps every existing construction site working unchanged.
    kind: str = "text"


class LlmModel:
    """Async facade over the proven Qwen leg + HF chat template.

    Fused path is mandatory (QwenFused batched, KV-cache aware) — no legacy
    fallback.
    """

    SIDECAR_BACKENDS = ("bonsai",)
    """Backends owning inference off-process (no local weights)."""

    @staticmethod
    def split_thinking(text: str) -> tuple:
        """Split model output into (thinking, answer).

        No tags -> ("", text); unclosed/orphan tags recovered.
        ..
        """
        raw = str(text or "")
        blocks = _THINK_RE.findall(raw)
        if blocks:
            thinking_parts = []
            for b in _THINK_RE.finditer(raw):
                g = b.group(0)
                thinking_parts.append(_THINK_CLOSE_RE.sub("", _THINK_OPEN_RE.sub("", g)).strip())
            thinking = "\n".join(p for p in thinking_parts if p).strip()
            answer = _THINK_RE.sub("", raw).strip()
            return thinking, answer
        o = _THINK_OPEN_RE.search(raw)
        c = _THINK_CLOSE_RE.search(raw)
        if o and not c:
            return _THINK_CLOSE_RE.sub("", raw[o.end():]).strip(), raw[:o.start()].strip()
        if c and not o:
            return raw[:c.start()].strip(), raw[c.end():].strip()
        return "", raw.strip()

    @staticmethod
    def leg_device(llm) -> str:
        """Where the warmed leg lives: cuda, cpu, unknown.

        Reports post-warm reality from the leg, not config intent.
        ..
        """
        leg = getattr(llm, "_leg", None)
        dev = getattr(leg, "device", None)
        if dev is not None:
            return str(dev)
        try:
            import torch

            model = getattr(leg, "model", leg)
            for param in model.parameters():
                return str(param.device)
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def assert_cuda_leg(llm, what: str = "model") -> str:
        """Abort unless the warmed leg is on CUDA. Returns device.

        Silent CPU fallback turns minutes into hours — fail fast.
        ..
        """
        dev = LlmModel.leg_device(llm)
        if not dev.startswith("cuda"):
            raise SystemExit(
                f"ABORT: {what} is on '{dev}', not CUDA. "
                f"Check nvidia-smi (driver/GPU state) and retry. "
                f"Refusing CPU fallback on purpose.")
        return dev

    @staticmethod
    def assert_ready_leg(llm, what: str = "model") -> str:
        """Backend-aware readiness gate. Returns a status string.

        Sidecars check health; GPU weight legs assert CUDA.
        ..
        """
        backend = str(getattr(getattr(llm, "config", None), "backend", ""))
        if backend in LlmModel.SIDECAR_BACKENDS:
            leg = llm._backend()
            url = getattr(leg, "base_url", "?")
            if not leg._health():
                raise SystemExit(
                    f"ABORT: {backend} sidecar unhealthy at {url}. "
                    f"Start llama-server first (see S0) and retry.")
            return f"sidecar {url}"
        return LlmModel.assert_cuda_leg(llm, what)

    @staticmethod
    def to_openai_tools(tools) -> list:
        """House specs -> OpenAI shape, idempotently. Pure.

        Already-converted specs pass through untouched either way.
        ..
        """
        out = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if isinstance(fn, dict) and "name" in fn:
                out.append(t)
                continue
            if "name" not in t:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object"}),
                },
            })
        return out

    @staticmethod
    def _tool_args(raw) -> dict:
        """Server arguments (dict or JSON str) -> args dict.

        Unparseable strings survive under "text", never dropped.
        ..
        """
        if isinstance(raw, dict):
            return dict(raw)
        try:
            parsed = json.loads(raw or "{}")
        except Exception:
            return {"text": str(raw or "")}
        return parsed if isinstance(parsed, dict) else {"text": str(raw or "")}

    @staticmethod
    def extract_tool_calls(text: str) -> tuple:
        """Template-native tool calls in order ((name, args), ...).

        Hermes <tool_call> blocks first (what the chat template emits
        when passed tools), bare OpenAI {"name", "arguments"} dicts as
        fallback. () means plain chat. Names pass through verbatim;
        mapping to execution lives outside this leg. Never raises.
        ..
        """
        if not text or not text.strip():
            return ()
        if "<tool_call>" in text.lower():
            out = []
            for m in _HERMES_RE.finditer(text):
                try:
                    obj = json.loads(m.group(1))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "") or "").strip()
                args = obj.get("arguments", {})
                if name and isinstance(args, dict):
                    out.append((name, args))
            if out:
                return tuple(out)
        hit = LlmModel._bare_dict(text)
        return (hit,) if hit is not None else ()

    @staticmethod
    def _bare_dict(text: str):
        """Bare OpenAI dict in text -> (name, args), else None.

        Balanced-brace scan from the first {"name" occurrence.
        ..
        """
        if not text or '"name"' not in text:
            return None
        try:
            start = text.index('{"name"')
        except ValueError:
            return None
        depth, instr, esc = 0, False, False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end <= start:
            return None
        try:
            obj = json.loads(text[start:end])
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        name = str(obj.get("name", "") or "").strip()
        args = obj.get("arguments", {})
        if not name or not isinstance(args, dict):
            return None
        return name, args

    def __init__(self, config: LlmConfig | None = None):
        self.config = config or LlmConfig()
        self._leg = None
        self._tok = None
        # Per-run preamble override (Sim-Agent subtasks). None = default.
        self.terminal_preamble = None

    def _backend(self):
        if self._leg is None:
            backend = getattr(self.config, "backend", "qwen") or "qwen"
            if backend == "bonsai":
                # 27B-ternary sidecar (Prism-fork llama-server, -ngl auto).
                # NOTE: resolved BEFORE any torch/CUDA probe — ngl probing
                # reads VRAM without initializing torch state.
                from src.models.bonsai import BonsaiLlamaCpp

                self._leg = BonsaiLlamaCpp(
                    gguf_path=getattr(self.config, "bonsai_gguf", "auto"),
                    packing=getattr(self.config, "bonsai_packing", "ptq1_0"),
                    mmproj=getattr(self.config, "bonsai_mmproj", "auto"),
                    port=getattr(self.config, "bonsai_port", 8081),
                    n_ctx=getattr(self.config, "bonsai_ctx", 16384),
                    n_gpu_layers=getattr(self.config, "bonsai_ngl", "auto"),
                    cache_type_k=getattr(self.config, "bonsai_cache_k",
                                         "q4_0"),
                    threads=getattr(self.config, "bonsai_threads", 0),
                    thinking_budget=getattr(self.config, "bonsai_thinking",
                                            None),
                    server_bin=getattr(self.config, "bonsai_bin", "auto"),
                )
                return self._leg
            from src.models.provider import ChatModel

            if ChatModel.is_cloud_backend(backend):
                raise SystemExit(
                    f"ABORT: backend={backend!r} is a stock LangChain cloud "
                    f"model now — use src/models/provider.ChatModel.get, "
                    f"not LlmModel."
                )
            import torch

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            # fused Qwen3 model: src/models/qwen.py (1 fused layer x28, batched, KV-cache)
            from src.models.qwen import QwenFused

            self._leg = QwenFused(
                device=device,
                model=self.config.model,
                max_seq=self.config.max_seq,
                max_new_tokens=self.config.max_tokens,
            )
        return self._leg

    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.config.model)
        return self._tok

    async def warm(self) -> "LlmModel":
        # Pre-execution guard BEFORE weights load: refuses a doomed warm
        # (wedged driver, second tenant, unfit VRAM/RAM) instead of hanging.
        from src.models.runtime.guard import (
            LEG_RAM_MB,
            LEG_VRAM_MB,
            preflight,
        )

        _be = str(getattr(self.config, "backend", ""))
        if _be in LlmModel.SIDECAR_BACKENDS:
            # Bonsai parks -ngl layers in VRAM: price them so the worker
            # cannot silently claim a full card. Pure-CPU sidecars skip
            # the GPU checks entirely.
            from src.models.runtime.guard import sidecar_gpu_estimate

            _need_gpu, _vram = sidecar_gpu_estimate(_be, self.config)
            preflight(f"llm warm [{_be}]", vram_mb=_vram,
                      needs_gpu=_need_gpu)
        else:
            preflight(f"llm warm [{_be}]",
                      vram_mb=LEG_VRAM_MB.get(_be, 2000.0),
                      ram_mb=LEG_RAM_MB.get(_be, 1500.0))
        # warm tokenizer first (lightweight) — except sidecar legs, which
        # need no local tokenizer (GGUF repos ship none; gated downloads
        # would fail here instead of at the leg with a clear message).
        if str(getattr(self.config, "backend", "")) not in LlmModel.SIDECAR_BACKENDS:
            await asyncio.to_thread(self._tokenizer)
        leg = await asyncio.to_thread(self._backend)
        if getattr(leg, "is_sidecar", False):
            # Sidecar legs own a server subprocess: construct isn't
            # enough — attach or spawn here so the first chat never
            # meets conn-refused.
            await asyncio.to_thread(leg.warm)
        return self

    def messages(self, text: str, history: list | None = None) -> list:
        """System + history + user turn in chat-template form."""
        msgs = [{"role": "system", "content": self.config.system_prompt}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": text})
        return msgs

    def messages_for_browser(self, text: str, history: list | None = None,
                             snapshot: str = "",
                             observation: str = "") -> list:
        """Same-model browser prompt: system + preamble + snapshot + user.

        No sidecar, no router model — the SAME Qwen weights decide between
        plain chat text and one JSON browser action. ``max_tokens`` default
        stays 48; callers pass a larger per-step limit to ``generate``
        (e.g. 128) so JSON fits.
        """
        # Prompt text lives in src.prompts.terminal (single home).
        from src.prompts.terminal import BROWSER_PREAMBLE as _BROWSER_PREAMBLE

        system = self.config.system_prompt + " " + _BROWSER_PREAMBLE
        msgs = [{"role": "system", "content": system}]
        msgs.extend(history or [])
        body = str(text or "")[:500]
        if snapshot:
            body += "\n" + str(snapshot)[:1500]
        if observation:
            body += "\nLast result: " + str(observation)[:500]
        msgs.append({"role": "user", "content": body})
        return msgs

    def messages_for_terminal(self, text: str, history: list | None = None,
                              cwd: str = "", observation: str = "",
                              catalogue: bool = True) -> list:
        """Same-model terminal prompt: system + preamble + cwd + user.

        No sidecar, no router model — the SAME Qwen weights decide between
        plain chat text and one template ``tool_call`` block (see
        src/models/chat_model.py). Keep bodies small so prompt+max_tokens
        fits the paged KV block pool. ``catalogue`` appends the prose op
        catalogue (for legs whose template may ignore native tool specs);
        native-function legs pass False — pure English suffices there.
        """
        from src.prompts.terminal import TERMINAL_PREAMBLE

        preamble = getattr(self, "terminal_preamble", None) or TERMINAL_PREAMBLE
        if catalogue:
            from src.prompts.terminal import TERMINAL_TOOL_CATALOGUE

            preamble = preamble + " " + TERMINAL_TOOL_CATALOGUE
        system = self.config.system_prompt + " " + preamble
        msgs = [{"role": "system", "content": system}]
        msgs.extend(history or [])
        body = str(text or "")[:500]
        if cwd:
            body += "\nCWD: " + str(cwd)[:300] + " SHELL: bash"
        if observation:
            body += "\nLast result: " + str(observation)[:800]
        msgs.append({"role": "user", "content": body})
        return msgs

    async def encode(self, messages: list, tools: list | None = None) -> list:
        tok = self._tokenizer()
        thinking = bool(self.config.thinking)

        def _run():
            kw: dict = {}
            if tools:
                kw["tools"] = LlmModel.to_openai_tools(tools)
            try:
                return tok.apply_chat_template(
                    messages, return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=thinking, **kw)["input_ids"][0].tolist()
            except TypeError:
                # Older tokenizer without the Qwen3 thinking switch (and
                # without tools support): plain template.
                try:
                    return tok.apply_chat_template(
                        messages, return_tensors="pt",
                        add_generation_prompt=True,
                        **kw)["input_ids"][0].tolist()
                except TypeError:
                    return tok.apply_chat_template(
                        messages, return_tensors="pt",
                        add_generation_prompt=True)["input_ids"][0].tolist()

        return await asyncio.to_thread(_run)

    async def decode(self, ids) -> str:
        """Decode token ids, preserving <think> traces.

        Thinking is NEVER stripped here: callers split via
        :func:`split_thinking` (terminal ``propose`` emits it as a
        ``thinking`` event, voice ``respond`` gates it from TTS, chat
        model forwards it in ``additional_kwargs``). Stripping here
        silently destroyed the think→toolcall loop (``decode_with_thinking``
        could never recover thinking once stripped).
        """
        tok = self._tokenizer()
        ids = [int(i) for i in ids]

        def _run():
            return tok.decode(ids, skip_special_tokens=True)

        return await asyncio.to_thread(_run)

    async def decode_with_thinking(self, ids) -> tuple:
        """Decode + split (thinking, answer). Never strips silently."""
        return LlmModel.split_thinking(await self.decode(ids))

    async def generate(self, messages: list,
                       max_tokens: int | None = None,
                       tools: list | None = None,
                       stop: list[str] | None = None) -> LlmResult:
        """Full reply (non-streaming). Raises when weights are missing.

        ``tools`` (OpenAI-style specs) is forwarded to the chat template;
        templates without tool support ignore it via encode() fallbacks.
        ``stop`` (e.g. ``["</function>"]``) is forwarded to legs with
        native stop support (MiniCPM HF) so a tool call ends cleanly at
        the envelope close; legs without it ignore the kwarg.
        """
        leg = self._backend()
        if getattr(leg, "is_sidecar", False):
            # CPU sidecar (Gemma): (messages, tools) straight through, no
            # ids round-trip (no local tokenizer for the GGUF repo).
            limit = int(max_tokens or self.config.max_tokens)
            out = await asyncio.to_thread(
                leg.chat, messages, tools, limit, stop)
            calls = []
            for tc in out.get("tool_calls", []) or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = str(fn.get("name", "") or "").strip()
                if not name:
                    continue
                calls.append((name, self._tool_args(fn.get("arguments", ""))))
            return LlmResult(text=out.get("text", ""),
                             ttft_s=float(out.get("ttft", 0.0)),
                             tps=float(out.get("decode_tps", 0.0)),
                             output_ids=(),
                             tool_calls=tuple(calls))
        ids = await self.encode(messages, tools=tools)
        limit = int(max_tokens or self.config.max_tokens)
        if stop:
            try:
                out = await asyncio.to_thread(
                    leg.generate, ids, limit, stop_strings=stop)
            except TypeError:
                # Leg without native stop support (fused/q4k): plain call.
                out = await asyncio.to_thread(
                    leg.generate, ids, int(max_tokens or self.config.max_tokens))
        else:
            out = await asyncio.to_thread(
                leg.generate, ids, int(max_tokens or self.config.max_tokens))
        # QwenFused returns {"ids": [..] } batched dict; handle both
        raw_ids = out["ids"] if isinstance(out, dict) else out
        if isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0] if len(raw_ids)==1 else raw_ids  # shouldn't happen for single
            if isinstance(raw_ids[0], list):
                raw_ids = raw_ids[0]
        text = await self.decode(raw_ids if isinstance(raw_ids, list) else list(raw_ids))
        n = len(raw_ids) if isinstance(raw_ids, (list, tuple)) else 0
        wall = float(out.get("ttft", 0.0)) + max(
            0.0, (n / max(float(out.get("decode_tps", 0.0)), 1e-9)) if n else 0.0) if isinstance(out, dict) else 0.0
        tps = (n / wall) if wall > 0 else 0.0
        ttft = float(out.get("ttft", 0.0)) if isinstance(out, dict) else 0.0
        return LlmResult(text=text, ttft_s=ttft,
                         tps=float(tps), output_ids=tuple(raw_ids if isinstance(raw_ids, (list,tuple)) else []),
                         tool_calls=self.extract_tool_calls(text))

    async def stream(self, messages: list, max_tokens: int | None = None,
                     tools: list | None = None,
                     stop: list[str] | None = None):
        """Async generator of :class:`LlmToken` as produced (for TTS/TUI).

        ``tools`` is forwarded to the chat template like in
        :meth:`generate`, so streamed terminal steps keep MiniCPM's native
        tool definitions (without them the streamed path would lose tool
        calls the non-streamed path makes). ``stop`` is forwarded to legs
        with native stop support so a tool call ends at the envelope.
        """
        import torch

        leg = self._backend()
        if getattr(leg, "is_sidecar", False):
            # CPU sidecar (Gemma): text pieces straight through; token_id
            # -1 tells callers to fall back to joined pieces (no local
            # token ids exist for the GGUF repo).
            limit = int(max_tokens or self.config.max_tokens)
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            acc: dict = {}

            def _run_sidecar():
                try:
                    for i, item in enumerate(
                            leg.chat_stream(acc, messages, tools, limit, stop)):
                        # Bonsai legs yield ("text"|"think", piece); older
                        # sidecars (Gemma) yield bare str == text.
                        if isinstance(item, (tuple, list)):
                            kind, piece = item[0], item[1]
                        else:
                            kind, piece = "text", item
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("tok", (kind, piece, i == 0)))
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                except Exception as exc:  # noqa: BLE001 - forwarded
                    loop.call_soon_threadsafe(queue.put_nowait, ("err", exc))

            worker = loop.run_in_executor(None, _run_sidecar)
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "end":
                        return
                    if kind == "err":
                        raise payload
                    pkind, piece, first = payload
                    yield LlmToken(token_id=-1, piece=str(piece),
                                   first=bool(first),
                                   kind=str(pkind or "text"))
            finally:
                await worker
            return

        ids = await self.encode(messages, tools=tools)
        limit = int(max_tokens or self.config.max_tokens)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                with torch.no_grad():
                    # QwenFused.generate returns dict, not stream; emulate stream via generate then chunk
                    # Prefer generate_stream if available (legacy QwenEngine), else fallback to generate
                    if hasattr(leg, "generate_stream"):
                        try:
                            it = (leg.generate_stream(ids, limit, stop_strings=stop)
                                  if stop else leg.generate_stream(ids, limit))
                        except TypeError:
                            # Leg without native stop support: plain stream.
                            it = leg.generate_stream(ids, limit)
                        for i, (tok_id, ttft) in enumerate(it):
                            loop.call_soon_threadsafe(
                                queue.put_nowait, ("tok", (tok_id, ttft, i == 0)))
                    else:
                        out = leg.generate(ids, limit)
                        raw = out["ids"] if isinstance(out, dict) else out
                        if isinstance(raw, list) and raw and isinstance(raw[0], list):
                            raw = raw[0]
                        for i, tok_id in enumerate(raw if isinstance(raw, list) else []):
                            loop.call_soon_threadsafe(queue.put_nowait, ("tok", (tok_id, 0.0, i == 0)))
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            except Exception as exc:  # noqa: BLE001 - forwarded to consumer
                loop.call_soon_threadsafe(queue.put_nowait, ("err", exc))

        worker = loop.run_in_executor(None, _run)
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    return
                if kind == "err":
                    raise payload
                tok_id, _ttft, first = payload
                piece = await self.decode([tok_id])
                yield LlmToken(token_id=int(tok_id), piece=piece,
                               first=bool(first))
        finally:
            await worker
