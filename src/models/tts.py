"""TTS leg: Kokoro synthesis behind a clean async class."""
import asyncio
from dataclasses import dataclass

import numpy as np

__all__ = ["TtsConfig", "TtsAudio", "TtsModel"]


@dataclass(frozen=True)
class TtsConfig:
    """No YAML: construct (or override fields) in code."""

    voice: str = "af_heart"
    lang: str = "a"
    sample_rate: int = 24000
    speed: float = 1.0
    device: str = "cuda"


@dataclass(frozen=True)
class TtsAudio:
    wav: np.ndarray = None  # float32 mono
    sample_rate: int = 24000
    sentence: str = ""
    synth_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self, "wav",
            np.asarray(self.wav if self.wav is not None else [],
                       dtype=np.float32).ravel())


class TtsModel:
    """Async facade over the proven Kokoro leg."""

    def __init__(self, config: TtsConfig | None = None):
        self.config = config or TtsConfig()
        self._leg = None

    def _backend(self):
        if self._leg is None:
            import torch

            # fused single-file model: src/models/kokoro.py (batched, VRAM-aware, fused tts kernels)
            from src.models.kokoro import KokoroFused

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._leg = KokoroFused(
                lang_code=self.config.lang,
                voice=self.config.voice,
                device=device,
                sample_rate=self.config.sample_rate,
                enhance=False,
                batch_size=4,
            )
        return self._leg

    async def warm(self) -> "TtsModel":
        def _run():
            leg = self._backend()
            try:
                leg.speak("warmup.")
            except Exception:
                pass
            return leg

        await asyncio.to_thread(_run)
        return self

    async def speak(self, text: str) -> TtsAudio:
        """Full text -> one audio chunk. Empty text -> empty audio."""
        import time as _time

        text = (text or "").strip()
        if not text:
            return TtsAudio(wav=np.zeros(0, dtype=np.float32),
                            sample_rate=self.config.sample_rate)
        leg = self._backend()
        t0 = _time.perf_counter()

        def _run():
            return leg.speak(text)

        wav, sr = await asyncio.to_thread(_run)
        return TtsAudio(wav=np.asarray(wav, dtype=np.float32),
                        sample_rate=int(sr), sentence=text,
                        synth_s=_time.perf_counter() - t0)

    async def speak_stream(self, sentences):
        """Iterable of sentences -> :class:`TtsAudio` per sentence."""
        for sent in sentences:
            if (sent or "").strip():
                yield await self.speak(sent)
