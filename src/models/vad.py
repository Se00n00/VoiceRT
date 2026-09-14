"""VAD leg: pure-ONNX speech detection behind a clean async class."""
import asyncio
from dataclasses import dataclass, field

import numpy as np

__all__ = ["VadConfig", "VadSegments", "VadModel"]


@dataclass(frozen=True)
class VadConfig:
    """No YAML: construct (or override fields) in code."""

    onnx_path: str = "src/models/engines/silero_vad/silero_vad.onnx"
    threshold: float = 0.5
    window: int = 512
    sample_rate: int = 16000
    min_speech_s: float = 0.25
    min_sil_s: float = 0.30


@dataclass(frozen=True)
class VadSegments:
    segments: tuple = field(default_factory=tuple)  # ((start_s, end_s), ...)
    audio_dur_s: float = 0.0

    @property
    def has_speech(self) -> bool:
        return len(self.segments) > 0

    @property
    def speech_s(self) -> float:
        return float(sum(e - s for s, e in self.segments))


class VadModel:
    """Async facade over the proven pure-ONNX Silero leg."""

    def __init__(self, config: VadConfig | None = None):
        self.config = config or VadConfig()
        self._leg = None

    def _backend(self):
        if self._leg is None:
            from src.models.engines.silero_vad.model import SileroVAD

            self._leg = SileroVAD(
                path=self.config.onnx_path,
                thresh=self.config.threshold,
                win=self.config.window,
                sr=self.config.sample_rate,
            )
        return self._leg

    async def warm(self) -> "VadModel":
        await asyncio.to_thread(self._backend)
        return self

    async def segments(self, audio, sr: int = 16000) -> VadSegments:
        """Full utterance -> speech spans. Raises when ONNX is unavailable."""
        leg = self._backend()
        cfg = self.config

        def _run():
            return leg.segment(
                np.asarray(audio, dtype=np.float32),
                sr=int(sr),
                min_speech_s=cfg.min_speech_s,
                min_sil_s=cfg.min_sil_s,
            )

        segs = await asyncio.to_thread(_run)
        dur = len(np.asarray(audio).ravel()) / float(sr or cfg.sample_rate)
        return VadSegments(
            segments=tuple((float(a), float(b)) for a, b in segs),
            audio_dur_s=float(dur),
        )

    async def active(self, chunk, sr: int = 16000) -> bool:
        """One streamed chunk -> speech present. Stateful; never raises."""
        try:
            leg = self._backend()
            frame = np.asarray(chunk, dtype=np.float32).ravel()
            if frame.size == 0:
                return False
            prob = await asyncio.to_thread(leg.prob, frame)
            return bool(prob >= self.config.threshold)
        except Exception:
            return False

    def reset(self) -> None:
        """Clear recurrent state at turn boundaries. Never raises."""
        try:
            if self._leg is not None:
                self._leg.reset()
        except Exception:
            pass
