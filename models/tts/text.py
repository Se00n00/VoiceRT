"""Text front-end for TTS: sentence splitting + phonemisation.

``espeak``/``misaki`` are imported lazily; without them :func:`phonemize`
falls back to plain normalised text so the pipeline still speaks.
"""
import re

__all__ = [
    "SPLIT",
    "split_sentences",
    "normalize_text",
    "phonemize",
]

# Abbreviation-aware sentence splitter (ported from VOICE/voice_engine.py).
SPLIT = re.compile(r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def split_sentences(text):
    """Split paragraphs into speakable sentences (non-empty, stripped)."""
    return [s.strip() for s in SPLIT.split(str(text)) if s.strip()]


def normalize_text(text):
    """Collapse whitespace and strip; real (if minimal) normalisation."""
    return _WS.sub(" ", str(text)).strip()


def phonemize(text, lang="en-us", backend="auto"):
    """Grapheme -> phoneme string.

    Tries ``misaki`` then ``espeak`` (both lazy); returns normalised plain
    text when neither is installed -- the Kokoro pipeline accepts that.
    """
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
