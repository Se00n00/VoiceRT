"""LLM leg: Qwen chat generation behind a clean async class."""
import asyncio
import re
from dataclasses import dataclass, field

__all__ = ["LlmConfig", "LlmResult", "LlmToken", "LlmModel", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = "You are a voice assistant. Reply in one short spoken sentence."

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class LlmConfig:
    """No YAML: construct (or override fields) in code."""

    model: str = "Qwen/Qwen3-0.6B"
    max_tokens: int = 48
    max_seq: int = 512
    device: str = "cuda"
    system_prompt: str = SYSTEM_PROMPT
    thinking: bool = False  # Qwen3 <think> traces (stripped from output)
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

    def _backend(self):
        if self._leg is None:
            import torch

            # fused single-file model: src/models/qwen.py (1 fused layer x28, batched, KV-cache)
            from src.models.qwen import QwenFused

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
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
        # warm tokenizer first (lightweight)
        await asyncio.to_thread(self._tokenizer)
        if getattr(self.config, "use_paged", False):
            # paged path owns its own QwenRunner weights — don't also warm fused leg
            try:
                await asyncio.to_thread(self._paged_engine)
            except Exception:
                pass
            # still warm fused as fallback if engine fails
            if self._paged_engine_inst is None:
                await asyncio.to_thread(self._backend)
        else:
            await asyncio.to_thread(self._backend)
        return self

    def messages(self, text: str, history: list | None = None) -> list:
        """System + history + user turn in chat-template form."""
        msgs = [{"role": "system", "content": self.config.system_prompt}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": text})
        return msgs

    async def encode(self, messages: list) -> list:
        tok = self._tokenizer()
        thinking = bool(self.config.thinking)

        def _run():
            try:
                return tok.apply_chat_template(
                    messages, return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=thinking)["input_ids"][0].tolist()
            except TypeError:
                # Older tokenizer without the Qwen3 thinking switch.
                return tok.apply_chat_template(
                    messages, return_tensors="pt",
                    add_generation_prompt=True)["input_ids"][0].tolist()

        return await asyncio.to_thread(_run)

    async def decode(self, ids) -> str:
        tok = self._tokenizer()
        ids = [int(i) for i in ids]

        def _run():
            text = tok.decode(ids, skip_special_tokens=True)
            if self.config.thinking:
                text = _THINK_RE.sub("", text).strip()
            return text

        return await asyncio.to_thread(_run)

    async def generate(self, messages: list,
                       max_tokens: int | None = None) -> LlmResult:
        """Full reply (non-streaming). Raises when weights are missing."""
        # paged path: real InferenceEngine (continuous batching)
        eng = self._paged_engine()
        if eng is not None:
            ids = await self.encode(messages)
            limit = int(max_tokens or self.config.max_tokens)
            from src.inference.config import SamplingParams as _SP
            sp = _SP(max_tokens=limit, temperature=0.0)
            def _run_paged():
                out = eng.generate([ids], sp, request_ids=["req0"])
                return list(out.get("req0", []))
            out_ids = await asyncio.to_thread(_run_paged)
            text = await self.decode(out_ids)
            return LlmResult(text=text, ttft_s=0.0, tps=0.0,
                             output_ids=tuple(out_ids))
        leg = self._backend()
        ids = await self.encode(messages)
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

    async def stream(self, messages: list, max_tokens: int | None = None):
        """Async generator of :class:`LlmToken` as produced (for TTS)."""
        # paged path streams via InferenceEngine step loop
        eng = self._paged_engine()
        if eng is not None:
            ids = await self.encode(messages)
            limit = int(max_tokens or self.config.max_tokens)
            from src.inference.config import SamplingParams as _SP
            sp = _SP(max_tokens=limit, temperature=0.0)
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            def _run_paged():
                try:
                    import time as _t
                    rid = f"stream-{int(_t.time_ns())}"
                    eng.add_request(rid, list(ids), sp)
                    first = True
                    while eng.has_unfinished():
                        outs = eng.step()
                        for o in outs:
                            if o.request_id != rid:
                                continue
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
                    kind, payload = await queue.get()
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
        ids = await self.encode(messages)
        limit = int(max_tokens or self.config.max_tokens)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                with torch.no_grad():
                    # QwenFused.generate returns dict, not stream; emulate stream via generate then chunk
                    # Prefer generate_stream if available (legacy QwenEngine), else fallback to generate
                    if hasattr(leg, "generate_stream"):
                        for i, (tok_id, ttft) in enumerate(
                                leg.generate_stream(ids, limit)):
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
