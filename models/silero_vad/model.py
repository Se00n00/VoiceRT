"""Silero VAD engine: ONNX session wrapper with owned chunking/segmentation.

Ported from ``VOICE/vad.py``. Expects 16 kHz mono audio. ``onnxruntime`` is
imported lazily so importing this module never crashes when it is missing;
in that case :meth:`SileroVAD.segments` falls back to the numpy energy VAD
in :mod:`models.silero_vad.kernels`.
"""
import numpy as np

from models.silero_vad import kernels as _kernels
from models.silero_vad.tokenizer import merge_segments

SR, WIN, CTX = 16000, 512, 64  # 32 ms windows + 64-sample rolling context

__all__ = ["SR", "WIN", "CTX", "SileroVAD"]


class SileroVAD:
    """Stateful Silero VAD wrapper.

    ``segment()`` is the primary entry point and returns
    ``[(start_s, end_s)]``; :meth:`segments` is kept as an alias.
    """

    def __init__(self, path=None, thresh=0.5, win=WIN):
        self.thresh = float(thresh)
        self.win = int(win)
        self.sr = SR
        self._sess = None
        self._sess_path = None
        self._ort_error = None
        self.path = path  # resolved lazily (may trigger a download)
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
            self._ort_error = exc
            if path is None:
                return None
            raise RuntimeError(
                "onnxruntime is required for SileroVAD (%r)" % (exc,))
        from models.silero_vad.weights import ensure_weights
        resolved = ensure_weights(path) if path is not None else ensure_weights()
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
        from runtime.tensor import to_host_numpy
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
        """Full utterance -> [(start_s, end_s)]. Falls back to energy VAD
        when onnxruntime/the weights are unavailable."""
        from runtime.tensor import to_host_numpy
        x = to_host_numpy(audio).ravel()
        if x.size == 0:
            return []
        try:
            sess = self._ensure_session(self.path)
        except Exception:
            sess = None
        if sess is None and self._sess is None:
            segs = _kernels.energy_segments(
                x, sr=sr, frame_len=self.win, thresh=self.thresh,
                min_speech_s=min_speech_s, min_sil_s=min_sil_s, pad_s=pad_s)
            return merge_segments(segs, max_gap_s=merge_gap_s or 0.0)
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
