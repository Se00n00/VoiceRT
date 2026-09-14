"""LLM leg: Qwen chat generation behind a clean async class."""
import asyncio
import re
import time
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
    """Async facade over the proven Qwen leg + HF chat template."""

    def __init__(self, config: LlmConfig | None = None):
        self.config = config or LlmConfig()
        self._leg = None
        self._tok = None

    def _backend(self):
        if self._leg is None:
            import torch

            from src.models.engines.qwen import QwenEngine

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._leg = QwenEngine(
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
        await asyncio.to_thread(self._backend)
        await asyncio.to_thread(self._tokenizer)
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
        leg = self._backend()
        ids = await self.encode(messages)
        out = await asyncio.to_thread(
            leg.generate, ids, int(max_tokens or self.config.max_tokens))
        text = await self.decode(out["ids"])
        n = len(out["ids"])
        wall = float(out.get("ttft", 0.0)) + max(
            0.0, (n / max(float(out.get("decode_tps", 0.0)), 1e-9)) if n else 0.0)
        tps = (n / wall) if wall > 0 else 0.0
        return LlmResult(text=text, ttft_s=float(out.get("ttft", 0.0)),
                         tps=float(tps), output_ids=tuple(out["ids"]))

    async def stream(self, messages: list, max_tokens: int | None = None):
        """Async generator of :class:`LlmToken` as produced (for TTS)."""
        import torch

        leg = self._backend()
        ids = await self.encode(messages)
        limit = int(max_tokens or self.config.max_tokens)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                with torch.no_grad():
                    for i, (tok_id, ttft) in enumerate(
                            leg.generate_stream(ids, limit)):
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("tok", (tok_id, ttft, i == 0)))
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
