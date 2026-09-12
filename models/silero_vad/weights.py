"""Download/cache helpers for the Silero VAD ONNX weights.

Canonical file is ``silero_vad.onnx`` (~2 MB, MIT) published with the
``snakers4/silero-vad`` torch.hub repo. We fetch it with stdlib ``urllib``
so this module has no heavy dependencies.
"""
import os
import urllib.request

REPO = "snakers4/silero-vad"
FILENAME = "silero_vad.onnx"
# Raw file backing the torch.hub ``snakers4/silero-vad`` package.
DEFAULT_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)
# Versioned fallback (v4.0 tag layout is identical for the onnx blob).
FALLBACK_URL = (
    "https://github.com/snakers4/silero-vad/raw/v4.0/src/silero_vad/data/"
    "silero_vad.onnx"
)

__all__ = [
    "REPO",
    "FILENAME",
    "DEFAULT_URL",
    "FALLBACK_URL",
    "default_cache_path",
    "is_cached",
    "download",
    "ensure_weights",
]


def default_cache_path():
    """Local cache path, honouring ``SILERO_VAD_PATH`` / ``HF_HUB_CACHE``."""
    env = os.environ.get("SILERO_VAD_PATH")
    if env:
        return env
    base = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "tinyinfer")
    return os.path.join(base, FILENAME)


def is_cached(path=None):
    """True when a non-empty onnx file exists at ``path`` (or the default)."""
    path = path or default_cache_path()
    return os.path.isfile(path) and os.path.getsize(path) > 0


def download(dest=None, url=None, timeout=120):
    """Fetch the onnx blob via urllib. Returns the destination path."""
    dest = dest or default_cache_path()
    urls = [url] if url else [DEFAULT_URL, FALLBACK_URL]
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    last_err = None
    for u in urls:
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(u, tmp)  # noqa: S310 (pinned hosts)
            if os.path.getsize(tmp) == 0:
                raise IOError("empty download from %s" % u)
            os.replace(tmp, dest)
            return dest
        except Exception as exc:  # try next mirror
            last_err = exc
    raise IOError("could not download silero_vad.onnx: %r" % (last_err,))


def ensure_weights(path=None, url=None):
    """Return a usable onnx path, downloading it on first use."""
    path = path or default_cache_path()
    if not is_cached(path):
        download(path, url=url)
    return path
