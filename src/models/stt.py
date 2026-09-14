"""STT leg: Whisper transcription behind a clean async class."""
import asyncio
from dataclasses import dataclass

import numpy as np

__all__ = ["SttConfig", "SttResult", "SttModel"]


@dataclass(frozen=True)
class SttConfig:
    """No YAML: construct (or override fields) in code."""

    model: str = "openai/whisper-base"
    language: str = "en"
    sample_rate: int = 16000
    max_tokens: int = 64
    device: str = "cuda"


@dataclass(frozen=True)
class SttResult:
    text: str = ""
    rtf: float = 0.0
    ttfs: float = 0.0
    dur_s: float = 0.0


class SttModel:
    """Async facade over the proven Whisper leg."""

    def __init__(self, config: SttConfig | None = None):
        self.config = config or SttConfig()
        self._leg = None
        self._proc = None

    def _backend(self):
        if self._leg is None:
            import torch

            from src.models.engines.whisper import WhisperEngine

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._leg = WhisperEngine(
                device=device,
                model=self.config.model,
                language=self.config.language,
                sample_rate=self.config.sample_rate,
                max_new_tokens=self.config.max_tokens,
            )
        return self._leg

    def _processor(self):
        if self._proc is None:
            from transformers import AutoProcessor

            self._proc = AutoProcessor.from_pretrained(self.config.model)
        return self._proc

    async def warm(self) -> "SttModel":
        await asyncio.to_thread(self._backend)
        await asyncio.to_thread(self._processor)
        return self

    async def transcribe(self, audio, sr: int = 16000) -> SttResult:
        """Waveform -> text. Raises on empty audio or missing weights."""
        import torch

        leg = self._backend()
        proc = self._processor()
        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        max_tokens = self.config.max_tokens

        def _run():
            feats = proc(wav, sampling_rate=int(sr),
                         return_tensors="pt").input_features
            feats = torch.nn.functional.pad(feats, (0, 3000 - feats.shape[-1]))
            if leg.device.startswith("cuda"):
                feats = feats.cuda()
            with torch.no_grad():
                r = leg.transcribe_mel(feats, max_tokens=max_tokens)
            ids = r["ids"]
            text = proc.batch_decode([ids], skip_special_tokens=True)[0]
            return r, text

        r, text = await asyncio.to_thread(_run)
        wall = float(r.get("total", 0.0))
        return SttResult(
            text=str(text),
            rtf=wall / max(dur_s, 1e-9),
            ttfs=float(r.get("ttfs", 0.0)),
            dur_s=float(dur_s),
        )
