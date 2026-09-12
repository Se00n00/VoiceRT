"""Voice / style-vector helpers for Kokoro TTS.

Kokoro ships per-voice style vectors (``af_heart`` etc.). Everything that
touches the ``kokoro`` package is lazy so importing this module is safe
without it.
"""
import os

__all__ = [
    "DEFAULT_VOICE",
    "KNOWN_VOICES",
    "LANG_CODE",
    "resolve_voice",
    "available_voices",
    "style_vector",
]

DEFAULT_VOICE = "af_heart"
LANG_CODE = "a"  # kokoro american-english pipeline code
KNOWN_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
)


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
    """Load a voice style vector tensor (lazy torch + kokoro data files).

    Returns a float tensor, or None when unavailable (caller then lets the
    pipeline use its own default packing).
    """
    voice = resolve_voice(voice)
    try:
        import torch
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
                return torch.load(c, map_location="cpu").float()
        except Exception:
            continue
    return None
