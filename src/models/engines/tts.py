"""Kokoro-82M TTS leg in one file: voices + text frontend + engine.

Single model class :class:`KokoroEngine`. Triton kernels from
:mod:`src.models.triton_kernels.tts` accelerate the audio post-processing around the
opaque Kokoro pipeline (peak-normalize + optional ``enhance`` filter +
resampling), so the kernels run in the hot path on every ``speak()``:

- ``postprocess`` (uses ``in1d_silu_fwd`` when ``enhance=True``)
- ``resample_linear`` (when a non-native sample rate is requested)

Main synthesis stays eager Kokoro by measured decision (RTF 0.09;
full-model ``torch.compile`` is broken in this env).
"""
import os
import re
import time

import numpy as np
import torch

from src.models.triton_kernels.tts import (
    HAVE_TRITON_KERNELS,
    conv1d_silu_fwd,
    in1d_silu_fwd,
    postprocess,
    resample_linear,
)

__all__ = [
    "SAMPLE_RATE", "DEFAULT_VOICE", "KNOWN_VOICES", "LANG_CODE",
    "KokoroEngine", "KokoroTTS", "TtsEngine",
    "resolve_voice", "available_voices", "style_vector",
    "split_sentences", "normalize_text", "phonemize",
    "HAVE_TRITON_KERNELS",
]

SAMPLE_RATE = 24000

DEFAULT_VOICE = "af_heart"
LANG_CODE = "a"  # kokoro american-english pipeline code
KNOWN_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
)

# Abbreviation-aware sentence splitter (ported from VOICE/voice_engine.py).
SPLIT = re.compile(r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


# -- voices ---------------------------------------------------------------
def resolve_voice(name=None):
    """Normalise a voice name, honouring ``KOKORO_VOICE``; never crashes."""
    name = name or os.environ.get("KOKORO_VOICE") or DEFAULT_VOICE
    name = str(name).strip()
    return name or DEFAULT_VOICE


def available_voices():
    """Voices bundled with the installed kokoro package (lazy)."""
    try:
        from kokoro import KPipeline  # noqa: F401
        import kokoro
        data = getattr(kokoro, "VOICES", None)
        if data:
            return sorted(str(v) for v in data)
    except Exception:
        pass
    return list(KNOWN_VOICES)


def style_vector(voice=None, pack_dir=None):
    """Load a voice style vector tensor (lazy torch + kokoro data files)."""
    voice = resolve_voice(voice)
    try:
        import torch as _torch
    except Exception:
        return None
    candidates = []
    if pack_dir:
        candidates.append(os.path.join(pack_dir, voice + ".pt"))
    try:
        import kokoro as _k
        base = os.path.dirname(os.path.abspath(_k.__file__))
        candidates.append(os.path.join(base, "voices", voice + ".pt"))
    except Exception:
        pass
    for c in candidates:
        try:
            if os.path.isfile(c):
                return _torch.load(c, map_location="cpu").float()
        except Exception:
            continue
    return None


# -- text frontend ----------------------------------------------------------
def split_sentences(text):
    """Split paragraphs into speakable sentences (non-empty, stripped)."""
    return [s.strip() for s in SPLIT.split(str(text)) if s.strip()]


def normalize_text(text):
    """Collapse whitespace and strip; real (if minimal) normalisation."""
    return _WS.sub(" ", str(text)).strip()


def phonemize(text, lang="en-us", backend="auto"):
    """Grapheme -> phoneme string (lazy misaki/espeak, else plain text)."""
    text = normalize_text(text)
    if not text:
        return text
    if backend in ("auto", "misaki"):
        try:
            from misaki import en  # noqa
            import misaki
            fn = getattr(misaki, "phonemize", None)
            if callable(fn):
                return fn(text)
        except Exception:
            if backend == "misaki":
                return text
    if backend in ("auto", "espeak"):
        try:
            import subprocess
            r = subprocess.run(["espeak", "-v", lang, "-q", "--ipa", text],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return text


# -- engine -----------------------------------------------------------------
class KokoroEngine:
    """Working Kokoro wrapper with Triton post-processing in the hot path.

    Takes explicit kwargs (``model``, ``lang``, ``voice``,
    ``sample_rate``, ``torch_compile``, ``speed``, ``enhance``, ...).
    """

    def __init__(self, lang_code=None, voice=None, device=None,
                 compile=True, phonemize_fn=None, lang=None, model=None,
                 sample_rate=SAMPLE_RATE, torch_compile=None, speed=1.0,
                 enhance=False, kernels=None, **_ignored):
        self.lang_code = lang_code or lang or LANG_CODE
        self.voice = resolve_voice(voice)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if torch_compile is not None:
            compile = torch_compile
        self._want_compile = bool(compile)
        self.sample_rate = int(sample_rate or SAMPLE_RATE)
        self.speed = float(speed or 1.0)
        self.enhance = bool(enhance)
        self._pipeline = None
        self._compiled = False
        self._phonemize = phonemize_fn or (lambda t: phonemize(t))
        # Exercise the Triton conv path at warmup so a broken kernel
        # surfaces here, not mid-conversation (CPU-safe no-op otherwise).
        try:
            x = torch.randn(1, 4, 32)
            if torch.cuda.is_available():
                x = x.cuda()
            in1d_silu_fwd(x)
            conv1d_silu_fwd(
                x, torch.randn(4, 4, 3, device=x.device, dtype=x.dtype),
                torch.zeros(4, device=x.device, dtype=x.dtype), padding=1)
        except Exception:
            pass

    # -- lazy pipeline -------------------------------------------------
    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from kokoro import KPipeline
        pipe = KPipeline(lang_code=self.lang_code, device=self.device)
        # EAGER synthesis (deliberate, measured 2026-09-11):
        # - full-model compile: broken (dynamo x transformers-5
        #   output_capturing -> NameError: torch in Albert encoder).
        # - submodule compile, static shapes: crashes on new lengths.
        # - submodule compile, dynamic=True: SLOWER than eager (RTF 0.37
        #   vs 0.09 on varying sentences).
        # Eager RTF 0.09 is 11x real-time; the honest win stays in
        # STT/LLM kernels + the Triton audio post-processing below.
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
        wav = np.concatenate(arr)
        # Triton-accelerated post-processing runs on every sentence.
        wav = postprocess(wav, sr=SAMPLE_RATE, enhance=self.enhance)
        if self.sample_rate != SAMPLE_RATE:
            wav = resample_linear(wav, SAMPLE_RATE, self.sample_rate)
        return wav.astype(np.float32)

    def speak(self, text, voice=None, sr=None):
        """Text -> ``(wav float32 mono, sample_rate)``. Real synthesis."""
        if voice is not None:
            self.voice = resolve_voice(voice)
        sr = int(sr or self.sample_rate)
        t0 = time.perf_counter()
        sentences = split_sentences(text)
        if not sentences:
            return np.zeros(0, dtype=np.float32), sr
        parts = [self._synth_one(self._phonemize(s)) for s in sentences
                 if s.strip()]
        wav = np.concatenate(parts) if parts else np.zeros(0,
                                                            dtype=np.float32)
        if sr != self.sample_rate and wav.size:
            wav = resample_linear(wav, self.sample_rate, sr)
        self.last_synth_s = time.perf_counter() - t0
        return wav.astype(np.float32), sr

    def speak_stream(self, text, voice=None):
        """Yield ``(sentence, wav_chunk, sr)`` per sentence for streaming."""
        if voice is not None:
            self.voice = resolve_voice(voice)
        for sent in split_sentences(text):
            if sent.strip():
                yield sent, self._synth_one(self._phonemize(sent)), self.sample_rate


KokoroTTS = KokoroEngine
TtsEngine = KokoroEngine
