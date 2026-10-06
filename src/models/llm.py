"""LLM leg: Qwen chat generation behind a clean async class."""
import asyncio
import re
from dataclasses import dataclass, field

__all__ = ["LlmConfig", "LlmResult", "LlmToken", "LlmModel", "SYSTEM_PROMPT",
           "split_thinking", "leg_device", "assert_cuda_leg",
           "assert_ready_leg", "SIDECAR_BACKENDS", "to_openai_tools"]

SYSTEM_PROMPT = "You are a voice assistant. Reply in one short spoken sentence."

# Backends served by a llama-server sidecar process instead of in-process
# weights. They own their own inference, so they need no local tokenizer,
# hold no VRAM we manage, and are incompatible with the paged GPU engine.
# Single source of truth: adding a sidecar backend means adding it HERE.
SIDECAR_BACKENDS = ("bonsai",)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def split_thinking(text: str) -> tuple:
    """Split model output into (thinking, answer).

    Robust: handles multiple blocks, mixed case, unclosed tags, and
    reasoning leaked outside tags. No tags -> ("", text). Pure, tested.
    """
    raw = str(text or "")
    # Collect all closed blocks
    blocks = _THINK_RE.findall(raw)
    if blocks:
        # Use first block's content as thinking, but join all if multiple
        thinking_parts = []
        for b in _THINK_RE.finditer(raw):
            g = b.group(0)
            thinking_parts.append(_THINK_CLOSE_RE.sub("", _THINK_OPEN_RE.sub("", g)).strip())
        thinking = "\n".join(p for p in thinking_parts if p).strip()
        answer = _THINK_RE.sub("", raw).strip()
        # If answer is empty but thinking exists, try to recover: maybe model
        # put answer inside think due to truncation — keep as thinking
        return thinking, answer
    o = _THINK_OPEN_RE.search(raw)
    c = _THINK_CLOSE_RE.search(raw)
    if o and not c:
        # Unclosed: everything after <think> is thinking
        return _THINK_CLOSE_RE.sub("", raw[o.end():]).strip(), raw[:o.start()].strip()
    if c and not o:
        # Orphan close: treat before as thinking
        return raw[:c.start()].strip(), raw[c.end():].strip()
    return "", raw.strip()


def leg_device(llm) -> str:
    """Where the warmed leg actually lives: 'cuda:0', 'cpu', or 'unknown'.

    Every backend sets ``.device`` (with CUDA fallback applied), so this
    reports post-warm reality, not config intent. Pure inspection.
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


def assert_cuda_leg(llm, what: str = "model") -> str:
    """Abort loudly unless the warmed leg is on CUDA. Returns device.

    A silent CPU fallback turns minutes into hours on this box — every
    runner calls this right after warm so a sick driver fails fast with
    a clear message instead of hanging.
    """
    dev = leg_device(llm)
    if not dev.startswith("cuda"):
        raise SystemExit(
            f"ABORT: {what} is on '{dev}', not CUDA. "
            f"Check nvidia-smi (driver/GPU state) and retry. "
            f"Refusing CPU fallback on purpose.")
    return dev


def assert_ready_leg(llm, what: str = "model") -> str:
    """Backend-aware readiness gate. Returns a status string.

    GPU weight legs go through :func:`assert_cuda_leg` (a sick driver
    fails fast); the bonsai sidecar (``backend="bonsai"``) must be
    healthy instead — they own their inference (llama-server), so the
    CUDA assert would wrongly abort them.
    """
    backend = str(getattr(getattr(llm, "config", None), "backend", ""))
    if backend in SIDECAR_BACKENDS:
        leg = llm._backend()
        url = getattr(leg, "base_url", "?")
        if not leg._health():
            raise SystemExit(
                f"ABORT: {backend} sidecar unhealthy at {url}. "
                f"Start llama-server first (see S0) and retry.")
        return f"sidecar {url}"
    return assert_cuda_leg(llm, what)


def to_openai_tools(tools) -> list:
    """House tool specs -> OpenAI ``{type, function}`` shape, idempotently.

    Every caller in this repo speaks the flat house shape
    (``{"name", "description", "parameters"}`` — see
    ``src.tools.terminal.TERMINAL_TOOLS``) because the sidecar legs convert
    it themselves on the way to ``/v1/chat/completions``. HF chat
    templates want the OpenAI shape instead, so the fused legs need this
    hop too. Already-converted specs pass through untouched, which lets
    callers hand us either form.
    """
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and "name" in fn:
            out.append(t)  # already OpenAI-shaped: no top-level name
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


@dataclass(frozen=True)
class LlmConfig:
    """No YAML: construct (or override fields) in code."""

    model: str = "Qwen/Qwen3-0.6B"
    # Backend switch.
    #  - "qwen" = fused Qwen3 path — DEFAULT: 0.6B fused on CUDA, ~1.4GB
    #    VRAM, no extra download on a box that already has the weights as
    #    the front leg. This is the same instance delegation routes chat to,
    #    so a bare library caller and the two-brain path agree on what the
    #    small model is. NOTE: a 0.6B is a chat/routing brain, not a tool
    #    harness — the agentic leg wants "bonsai" below.
    #  - "bonsai" = Ternary Bonsai 2 27B via Prism-fork llama.cpp sidecar
    #    (-ngl auto from free VRAM, 16K ctx) — the WORKER brain for the
    #    full tool harness; STT/TTS keep the GPU, the rest offloads.
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
    # paged inference engine (real QwenRunner, no dummy). Disabled by default
    # so tests stay fast; enable in VoiceAgent/server for batching.
    use_paged: bool = False
    paged_blocks: int = 16
    paged_batch_size: int = 4
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_cuda_graph: bool = False


@dataclass(frozen=True)
class LlmResult:
    text: str = ""
    ttft_s: float = 0.0
    tps: float = 0.0
    output_ids: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class LlmToken:
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
    fallback. Optional paged path via :mod:`src.inference` when
    ``config.use_paged`` is True (real QwenRunner, paged KV, continuous
    batching, prefix cache / chunked / CUDA-graph).
    """

    def __init__(self, config: LlmConfig | None = None):
        self.config = config or LlmConfig()
        self._leg = None
        self._tok = None
        self._paged_engine_inst = None
        # Per-run preamble override (Sim-Agent subtasks). None = default.
        self.terminal_preamble = None

    def _backend(self):
        if self._leg is None:
            backend = getattr(self.config, "backend", "qwen") or "qwen"
            if backend == "bonsai":
                # 27B-ternary sidecar (Prism-fork llama-server, -ngl auto).
                # NOTE: resolved BEFORE any torch/CUDA probe — ngl probing
                # reads VRAM without initializing torch state.
                from src.models.bonsai_llamacpp import BonsaiLlamaCpp

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

    def _paged_engine(self):
        """Lazily build paged InferenceEngine (real QwenRunner, no dummy)."""
        if not getattr(self.config, "use_paged", False):
            return None
        if str(getattr(self.config, "backend", "")) in SIDECAR_BACKENDS:

            # Sidecar legs own their inference (llama-server); the paged
            # GPU engine is incoherent here — and must never warm GPU
            # weights behind a sidecar backend's back.
            return None
        if self._paged_engine_inst is not None:
            return self._paged_engine_inst
        try:
            from src.inference import EngineConfig, InferenceEngine
            import torch
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            # auto-size blocks from VRAM but keep small for 4GB demo
            cfg = EngineConfig(
                model=self.config.model,
                device=device,
                max_seq_len=self.config.max_seq,
                num_blocks=getattr(self.config, "paged_blocks", 0) or 16,
                max_batch_size=getattr(self.config, "paged_batch_size", 4),
                default_max_tokens=self.config.max_tokens,
                enable_prefix_caching=getattr(self.config, "enable_prefix_caching", True),
                enable_chunked_prefill=getattr(self.config, "enable_chunked_prefill", True),
                enable_cuda_graph=getattr(self.config, "enable_cuda_graph", False),
                enable_fused_attention=True,
            )
            eng = InferenceEngine(cfg, device=device, runner="qwen")
            if not getattr(eng.runner, "loaded", False):
                return None
            self._paged_engine_inst = eng
            return eng
        except Exception:
            return None

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
        if _be in SIDECAR_BACKENDS:
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
        if str(getattr(self.config, "backend", "")) not in SIDECAR_BACKENDS:
            await asyncio.to_thread(self._tokenizer)
        if getattr(self.config, "use_paged", False):
            # paged path owns its own QwenRunner weights — don't also warm fused leg
            try:
                await asyncio.to_thread(self._paged_engine)
            except Exception:
                pass
            # still warm fused as fallback if engine fails
            if self._paged_engine_inst is None:
                leg = await asyncio.to_thread(self._backend)
                if getattr(leg, "is_sidecar", False):
                    await asyncio.to_thread(leg.warm)
        else:
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
        # Inlined (was src.tools.schema.BROWSER_PREAMBLE): the browser
        # schema module was removed with the extension; the prompt text
        # stays here so this helper keeps working stand-alone.
        _BROWSER_PREAMBLE = (
            "You control a browser. Reply with EITHER plain chat text "
            "OR exactly one JSON action. No other text when acting. "
            'Ops: click {action,ref} | type {action,ref,text} | '
            'scroll {action,ref,direction} | select {action,ref,text} | '
            'navigate {action,text:url} | read {action,ref?} | '
            'done {action,reply}. Use refs from the page list only.'
        )

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
                              cwd: str = "", observation: str = "") -> list:
        """Same-model terminal prompt: system + preamble + cwd + user.

        No sidecar, no router model — the SAME Qwen weights decide between
        plain chat text and one JSON ``TerminalAction`` (see
        src/tools/terminal.py). Keep bodies small so prompt+max_tokens
        fits the paged KV block pool.
        """
        from src.agent.prompts import TERMINAL_PREAMBLE

        preamble = getattr(self, "terminal_preamble", None) or TERMINAL_PREAMBLE
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
                kw["tools"] = to_openai_tools(tools)
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
        return split_thinking(await self.decode(ids))

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
        # paged path: real InferenceEngine (continuous batching)
        eng = self._paged_engine()
        if eng is not None:
            ids = await self.encode(messages, tools=tools)
            limit = int(max_tokens or self.config.max_tokens)
            from src.inference.config import SamplingParams as _SP
            sp = _SP(max_tokens=limit, temperature=0.0)
            # unique id per call: a stuck/timed-out call must never poison
            # later ones with "duplicate request_id" (fixed "req0" did).
            import time as _t
            rid = f"req-{int(_t.time_ns())}"
            def _run_paged():
                out = eng.generate([ids], sp, request_ids=[rid])
                return list(out.get(rid, []))
            out_ids = await asyncio.to_thread(_run_paged)
            text = await self.decode(out_ids)
            return LlmResult(text=text, ttft_s=0.0, tps=0.0,
                             output_ids=tuple(out_ids))
        leg = self._backend()
        if getattr(leg, "is_sidecar", False):
            # CPU sidecar (Gemma): (messages, tools) straight through, no
            # ids round-trip (no local tokenizer for the GGUF repo).
            limit = int(max_tokens or self.config.max_tokens)
            out = await asyncio.to_thread(
                leg.chat, messages, tools, limit, stop)
            return LlmResult(text=out.get("text", ""),
                             ttft_s=float(out.get("ttft", 0.0)),
                             tps=float(out.get("decode_tps", 0.0)),
                             output_ids=())
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
                         tps=float(tps), output_ids=tuple(raw_ids if isinstance(raw_ids, (list,tuple)) else []))

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
        # paged path streams via InferenceEngine step loop
        eng = self._paged_engine()
        if eng is not None:
            ids = await self.encode(messages, tools=tools)
            limit = int(max_tokens or self.config.max_tokens)
            from src.inference.config import SamplingParams as _SP
            sp = _SP(max_tokens=limit, temperature=0.0)
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            import time as _t
            rid = f"stream-{int(_t.time_ns())}"
            empty_steps = {"n": 0}  # worker-side stall counter (sees step outs)
            def _run_paged():
                try:
                    eng.add_request(rid, list(ids), sp)
                    first = True
                    while eng.has_unfinished():
                        outs = eng.step()
                        if not outs:
                            # Nothing scheduled for anyone: a few empties are
                            # normal at admission; hundreds mean the prompt can
                            # never fit the block pool (livelock).
                            empty_steps["n"] += 1
                            if empty_steps["n"] >= 500:
                                try:
                                    eng.abort_request(rid)
                                except Exception:
                                    pass
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, ("err", RuntimeError(
                                        "inference stalled: prompt likely exceeds KV block pool; "
                                        "shorten history/snapshot or raise paged blocks")))
                                return
                            continue
                        empty_steps["n"] = 0
                        mine = [o for o in outs if o.request_id == rid]
                        for o in mine:
                            loop.call_soon_threadsafe(queue.put_nowait, ("tok", (o.token_id, first)))
                            first = False
                            if o.finished:
                                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                                return
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("err", exc))
            worker = loop.run_in_executor(None, _run_paged)
            try:
                while True:
                    try:
                        # Stall backstop: worker only posts tokens; silence
                        # here means it is wedged — abort and fail loudly.
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=180)
                    except asyncio.TimeoutError:
                        try:
                            eng.abort_request(rid)
                        except Exception:
                            pass
                        raise RuntimeError("LLM stream timed out (180s without a token)")
                    if kind == "end":
                        return
                    if kind == "err":
                        raise payload
                    tok_id, first = payload
                    piece = await self.decode([tok_id])
                    yield LlmToken(token_id=int(tok_id), piece=piece, first=bool(first))
            finally:
                await worker
            return

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
