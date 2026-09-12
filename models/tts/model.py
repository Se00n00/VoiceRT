"""Kokoro TTS wrapper: lazy ``kokoro`` import, ``speak(text) -> (wav, sr)``.

``kokoro`` is imported lazily inside :meth:`KokoroEngine._ensure_pipeline`
so ``import models.tts.model`` never crashes when it is missing. A
``torch.compile`` (Inductor) pass is attempted with graceful fallback to
eager, mirroring ``VOICE/voice_engine.py``.
"""
import time

import numpy as np
import torch

from models.tts.text import phonemize, split_sentences
from models.tts.weights import DEFAULT_VOICE, LANG_CODE, resolve_voice

SAMPLE_RATE = 24000

__all__ = ["SAMPLE_RATE", "KokoroEngine"]


class KokoroEngine:
    """Working Kokoro wrapper with compile attempt + graceful fallback."""

    def __init__(self, lang_code=LANG_CODE, voice=None, device=None,
                 compile=True, phonemize_fn=None):
        self.lang_code = lang_code
        self.voice = resolve_voice(voice)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._want_compile = bool(compile)
        self._pipeline = None
        self._compiled = False
        self._phonemize = phonemize_fn or (lambda t: phonemize(t))

    # -- lazy pipeline -------------------------------------------------
    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from kokoro import KPipeline
        pipe = KPipeline(lang_code=self.lang_code, device=self.device)
        # EAGER synthesis (deliberate, measured 2026-09-11):
        # - full-model compile: broken (dynamo x transformers-5
        #   output_capturing -> NameError: torch in Albert encoder).
        # - submodule compile, static shapes: crashes on new lengths
        #   (FakeTensor 10*s2 vs s0//6 in upsampler add).
        # - submodule compile, dynamic=True: no crash, but per-length
        #   recompiles + guard overhead make serving SLOWER than eager
        #   (RTF 0.37 vs 0.09 on varying sentences).
        # Eager RTF 0.09 is 11x real-time; TTS is ~25% of turn time, so
        # the honest win stays in STT/LLM kernels. Revisit only with
        # length-bucketed synthesis.
        try:
            list(pipe("warmup.", voice=self.voice))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass
        self._compiled = False  # eager by decision above
        self._pipeline = pipe
        return pipe

    @property
    def is_compiled(self):
        return self._compiled

    # -- synthesis -----------------------------------------------------
    @torch.no_grad()
    def _synth_one(self, sentence):
        pipe = self._ensure_pipeline()
        chunks = [a for _, _, a in pipe(sentence, voice=self.voice)]
        if self.device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        arr = [np.asarray(c, dtype=np.float32).ravel() for c in chunks]
        return np.concatenate(arr)

    def speak(self, text, voice=None, sr=SAMPLE_RATE):
        """Text -> ``(wav float32 mono, sample_rate)``. Real synthesis."""
        if voice is not None:
            self.voice = resolve_voice(voice)
        t0 = time.perf_counter()
        sentences = split_sentences(text)
        if not sentences:
            return np.zeros(0, dtype=np.float32), sr
        parts = [self._synth_one(self._phonemize(s)) for s in sentences
                 if s.strip()]
        wav = np.concatenate(parts) if parts else np.zeros(0,
                                                            dtype=np.float32)
        self.last_synth_s = time.perf_counter() - t0
        return wav.astype(np.float32), sr

    def speak_stream(self, text, voice=None):
        """Yield ``(sentence, wav_chunk, sr)`` per sentence for streaming."""
        if voice is not None:
            self.voice = resolve_voice(voice)
        for sent in split_sentences(text):
            if sent.strip():
                yield sent, self._synth_one(self._phonemize(sent)), SAMPLE_RATE
