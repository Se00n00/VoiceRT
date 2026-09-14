"""Silero VAD engine: pure-ONNX session wrapper with owned segmentation.

Expects 16 kHz mono audio. Deliberately the whole leg: VAD is a tiny
(~2 MB) ONNX model running on CPU at RTF ~0.01, so there is no numpy
energy fallback and no Triton path. When onnxruntime or the weights are
unavailable the leg raises (omit-and-report: callers keep it as None
and surface it in ``missing`` instead of silently degrading).

Config aliases: takes ``onnx_path``/``threshold``/``window``/
``sample_rate`` (plus ``path``/``thresh``/``win``/``sr``) as kwargs;
anything else is ignored.
"""
import os

import numpy as np

from src.models.engines.silero_vad.tokenizer import merge_segments

SR, WIN, CTX = 16000, 512, 64  # 32 ms windows + 64-sample rolling context

__all__ = ["SR", "WIN", "CTX", "SileroVAD"]


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(here))))


def _looks_like_path(value):
    return (isinstance(value, str) and bool(value)
            and (value.endswith(".onnx") or os.path.sep in value
                 or os.path.isfile(value)))


def _vendored_weights():
    cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "silero_vad.onnx")
    if os.path.isfile(cand) and os.path.getsize(cand) > 0:
        return cand
    return None


class SileroVAD:
    """Stateful Silero VAD wrapper.

    ``segment()`` is the primary entry point and returns
    ``[(start_s, end_s)]``; :meth:`segments` is kept as an alias.
    :meth:`prob` scores one frame through the stateful LSTM session and
    is what streaming callers (WS endpointing) use per chunk.
    """

    def __init__(self, path=None, thresh=0.5, win=WIN, sr=SR, **ignored):
        if path is None:
            path = ignored.get("onnx_path")
        if not _looks_like_path(path):
            path = None
        thresh = ignored.get("threshold", thresh)
        win = ignored.get("window", win)
        sr = ignored.get("sample_rate", sr)
        self.thresh = float(thresh)
        self.win = int(win)
        self.sr = int(sr)
        self._sess = None
        self._sess_path = None
        self.path = path  # resolved lazily (vendored file, then download)
        self.reset()
        if path is not None:
            self._ensure_session(path)

    # -- session handling (lazy onnxruntime) ---------------------------
    def _ensure_session(self, path=None):
        if self._sess is not None:
            return self._sess
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is required for SileroVAD (%r)" % (exc,))
        candidate = path or self.path
        if not _looks_like_path(candidate):
            candidate = None
        if candidate is not None and not os.path.isabs(candidate):
            candidate = os.path.join(_repo_root(), candidate)
        if candidate is None or not os.path.isfile(candidate):
            candidate = _vendored_weights() or candidate
        from src.models.engines.silero_vad.weights import ensure_weights
        resolved = (candidate if candidate is not None
                    and os.path.isfile(candidate)
                    else ensure_weights(candidate))
        self._sess = ort.InferenceSession(
            resolved, providers=["CPUExecutionProvider"])
        self._sess_path = resolved
        self.path = resolved
        return self._sess

    @property
    def using_onnx(self):
        return self._sess is not None

    def reset(self):
        """Clear recurrent state + rolling context."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CTX), dtype=np.float32)

    # -- frame-level scoring -------------------------------------------
    def prob(self, frame):
        """One 32 ms frame -> speech probability. Stateful."""
        sess = self._ensure_session(self.path)
        from src.models.runtime.tensor import to_host_numpy
        x = to_host_numpy(frame).reshape(1, -1)
        if x.shape[1] < self.win:
            x = np.pad(x, ((0, 0), (0, self.win - x.shape[1])))
        x = np.concatenate([self.context, x[:, :self.win]], axis=1)
        self.context = x[:, -CTX:]
        out, self.state = sess.run(
            None, {"input": x, "state": self.state,
                   "sr": np.array(SR, dtype=np.int64)})
        return float(np.asarray(out).flat[0])

    # -- utterance-level segmentation ----------------------------------
    def segment(self, audio, sr=SR, min_speech_s=0.25, min_sil_s=0.3,
                pad_s=0.03, merge_gap_s=0.0):
        """Full utterance -> [(start_s, end_s)]. Requires the ONNX session."""
        from src.models.runtime.tensor import to_host_numpy
        x = to_host_numpy(audio).ravel()
        if x.size == 0:
            return []
        self._ensure_session(self.path)
        self.reset()
        n = len(x) // self.win
        if n == 0:
            p = self.prob(x)
            return [(0.0, len(x) / sr)] if p >= self.thresh else []
        probs = [self.prob(x[i * self.win:(i + 1) * self.win])
                 for i in range(n)]
        segs, start, sil = [], None, 0
        need_sil = max(1, int(round(min_sil_s / (self.win / sr))))
        for i, p in enumerate(probs):
            t = i * self.win / sr
            if p >= self.thresh:
                if start is None:
                    start = max(0.0, t - pad_s)
                sil = 0
            elif start is not None and (t - start) >= min_speech_s:
                sil += 1
                if sil >= need_sil:
                    segs.append((start, t + pad_s))
                    start, sil = None, 0
        if start is not None:
            segs.append((start, len(x) / sr))
        if merge_gap_s:
            segs = merge_segments(segs, max_gap_s=merge_gap_s)
        return segs

    # Alias kept for the original ``VOICE/vad.py`` call sites.
    def segments(self, audio, **kwargs):
        return self.segment(audio, **kwargs)
