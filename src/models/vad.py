"""VAD leg: Silero speech detection via the pip ``silero-vad`` package.

Single-file facade: weights ship inside the installed wheel
(``silero-vad==6.2.3``, see requirements.txt), so there is no vendored
onnx blob and no engine module. ``onnx=True`` (default) uses the pip
ONNX backend — byte-identical weights to the old vendored file, fastest
on CPU (~0.12ms/frame); ``onnx=False`` uses the torch JIT twin.


Audio ~ VAD --> 
[
    {
        'start': 0.3, 
        'end': 2.9
    }, {
        'start': 3.5, 
        'end': 7.2
    }
]
"""
import asyncio
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

_TORCH_THREADS = torch.get_num_threads()

from silero_vad import get_speech_timestamps, load_silero_vad

torch.set_num_threads(_TORCH_THREADS)  # undo pip's global clamp to 1

from src.models.runtime.guard import VAD_RAM_MB, preflight

_SR = 16000
_WIN = 512


class VadConfig(BaseModel):
    """No YAML: construct (or override fields) in code."""

    model_config = ConfigDict(frozen=True)

    threshold: float = 0.5
    window: int = _WIN
    sample_rate: int = _SR
    min_speech_s: float = 0.25
    min_sil_s: float = 0.30
    onnx: bool = True  # pip backend: True = ONNX (fast), False = torch JIT


class VadSegments(BaseModel):
    model_config = ConfigDict(frozen=True)

    segments: tuple = Field(default_factory=tuple)  # ((start_s, end_s), ...)
    audio_dur_s: float = 0.0

    @property
    def has_speech(self) -> bool:
        return len(self.segments) > 0

    @property
    def speech_s(self) -> float:
        return float(sum(e - s for s, e in self.segments))


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample to 16 kHz. Pass-through when already 16 kHz."""
    if int(sr) == _SR:
        return x
    n_out = max(1, int(len(x) * _SR / int(sr)))
    xp = np.linspace(0.0, 1.0, len(x))
    return np.interp(np.linspace(0.0, 1.0, n_out), xp, x).astype(np.float32)


def _windows(x: np.ndarray, n: int):
    """Yield ``n``-sample windows, zero-padding the tail."""
    for i in range(0, len(x), n):
        w = x[i:i + n]
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)))
        yield w


class VadModel:
    """Async facade over the pip Silero VAD leg."""

    def __init__(self, config: VadConfig | None = None):
        self.config = config or VadConfig()
        self._leg: Any = None

    def _backend(self):
        if self._leg is None:
            self._leg = load_silero_vad(onnx=self.config.onnx)
            self._leg.reset_states()
        return self._leg

    def _prob(self, window: np.ndarray) -> float:
        """One 512-sample @16k window -> speech probability. Stateful."""
        out = self._backend()(torch.from_numpy(window), _SR)
        return float(out.item() if isinstance(out, torch.Tensor)
                     else np.asarray(out).flat[0])

    async def warm(self) -> "VadModel":
        preflight("vad warm", needs_gpu=False, ram_mb=VAD_RAM_MB)

        def _run():
            leg = self._backend()
            leg(torch.from_numpy(np.zeros(_WIN, dtype=np.float32)), _SR)
            return leg

        await asyncio.to_thread(_run)
        return self

    async def segments(self, audio, sr: int = 16000) -> VadSegments:
        """Full utterance -> speech spans. Raises on backend failure."""
        cfg = self.config
        x = np.asarray(audio, dtype=np.float32).ravel()
        dur = len(x) / float(sr or cfg.sample_rate)
        if x.size == 0:
            return VadSegments(segments=(), audio_dur_s=0.0)
        wav16 = _resample_to_16k(x, int(sr or cfg.sample_rate))

        def _run():
            return get_speech_timestamps(
                torch.from_numpy(wav16), self._backend(),
                threshold=cfg.threshold,
                sampling_rate=_SR,
                min_speech_duration_ms=int(cfg.min_speech_s * 1000),
                min_silence_duration_ms=int(cfg.min_sil_s * 1000),
                speech_pad_ms=30,
                return_seconds=True,
            )

        raw = await asyncio.to_thread(_run)
        return VadSegments(
            segments=tuple((float(s["start"]), float(s["end"])) for s in raw),
            audio_dur_s=float(dur),
        )

    async def active(self, chunk, sr: int = 16000) -> bool:
        """One streamed chunk -> speech present. Stateful; never raises."""
        try:
            x = np.asarray(chunk, dtype=np.float32).ravel()
            if x.size == 0:
                return False
            wav16 = _resample_to_16k(x, int(sr))

            def _run():
                return any(self._prob(w) >= self.config.threshold
                           for w in _windows(wav16, self.config.window))

            return bool(await asyncio.to_thread(_run))
        except Exception:
            return False

    def reset(self) -> None:
        """Clear recurrent state at turn boundaries. Never raises."""
        try:
            if self._leg is not None:
                self._leg.reset_states()
        except Exception:
            pass
