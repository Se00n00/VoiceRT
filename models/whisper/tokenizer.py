"""Whisper token helpers: id <-> text via HF transformers (lazy) with a
minimal byte-level fallback decoder so text recovery works offline.
"""
import re

__all__ = [
    "SOT",
    "EOT",
    "MODEL_ID",
    "decode_ids",
    "encode_text",
    "clean_text",
    "byte_fallback_decode",
]

SOT = 50258  # <|startoftranscript|> for whisper-base
EOT = 50257  # <|endoftext|>
MODEL_ID = "openai/whisper-base"
_SPECIAL = re.compile(r"<\|[^|]*\|>")


def _get_processor(model_id=MODEL_ID):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


def decode_ids(ids, model_id=MODEL_ID, skip_special_tokens=True):
    """Token ids -> text. Lazy transformers; byte-fallback offline."""
    ids = [int(i) for i in ids]
    try:
        proc = _get_processor(model_id)
        return proc.batch_decode([ids],
                                 skip_special_tokens=skip_special_tokens)[0]
    except Exception:
        return byte_fallback_decode(ids)


def encode_text(text, model_id=MODEL_ID):
    """Text -> prompt token ids (lazy transformers)."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    tok = proc.tokenizer
    return tok(text, return_tensors="pt").input_ids[0].tolist()


def clean_text(text):
    """Strip ``<|...|>`` specials and surrounding whitespace."""
    return _SPECIAL.sub("", text).strip()


def byte_fallback_decode(ids):
    """Minimal GPT-2 byte-level decode for offline use (real code).

    Reconstructs the byte string for printable ASCII/UTF-8 byte tokens and
    drops special token ids (>= 50257) when they carry no text.
    """
    try:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    except Exception:
        return clean_text(" ".join(
            "<|%d|>" % i if i >= 50257 else chr(i % 128) for i in ids))
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    out = bytearray()
    for i in ids:
        if i >= 50257:
            continue
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
            piece = tok.decode([i], skip_special_tokens=False)
        except Exception:
            piece = ""
        for ch in piece:
            if ch in u2b:
                out.append(u2b[ch])
    try:
        return clean_text(out.decode("utf-8", errors="ignore"))
    except Exception:
        return ""
