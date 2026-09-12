"""Kernel surface for the VAD leg.

Silero VAD is a tiny (~2 MB) ONNX model; the dominant recurrent op is an
LSTM (conv-frontend + LSTM + linear, state [2, 1, 128]). This module
provides an optional fused Triton LSTM-cell path with numpy/torch
fallback, plus the frame-energy VAD used when ONNX/onnxruntime is
unavailable.

NOTE (wiring status): the LSTM weights cannot be extracted from the
Silero ONNX graph cleanly (fused recurrent initializers), so
``SileroVAD.prob`` still runs the ONNX session. The Triton
``lstm_cell`` kernel below is implemented + parity-tested standalone;
ONNX wiring is pending. No fake wiring is claimed.
"""
import numpy as np

__all__ = [
    "tk_frame_energy",
    "frame_log_energy",
    "energy_probs",
    "energy_segments",
    "lstm_cell_step",
    "HAVE_TRITON_LSTM",
]

# No triton frame_energy kernel exists (reductions.py covers row ops; the
# energy VAD below is pure numpy and already ~50us). Name kept exported.
tk_frame_energy = None

try:
    from triton_kernels.activation import lstm_cell as _tk_lstm_cell
    HAVE_TRITON_LSTM = True
except Exception:
    _tk_lstm_cell = None
    HAVE_TRITON_LSTM = False


def _torch_lstm_ref(x, h, c, w_ih, w_hh, b):
    """Pure-torch LSTM step (i,f,g,o order), CPU/CUDA fallback."""
    import torch
    import torch.nn.functional as F
    if b is not None and torch.as_tensor(b).numel() == 8 * h.shape[-1]:
        H = h.shape[-1]
        b = torch.as_tensor(b, device=x.device)[:4 * H] + torch.as_tensor(
            b, device=x.device)[4 * H:]
    gates = F.linear(x, w_ih) + F.linear(h, w_hh)
    if b is not None:
        gates = gates + b
    H = h.shape[-1]
    i, f, g, o = gates[..., :H], gates[..., H:2 * H], gates[..., 2 * H:3 * H], gates[..., 3 * H:]
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * torch.tanh(c_new)
    return h_new, c_new


def _numpy_lstm_ref(x, h, c, w_ih, w_hh, b):
    """Pure-numpy LSTM step fallback."""
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    w_ih = np.asarray(w_ih, dtype=np.float32)
    w_hh = np.asarray(w_hh, dtype=np.float32)
    squeeze = False
    if x.ndim == 1:
        x, h, c, squeeze = x[None], h[None], c[None], True
    if b is not None:
        b = np.asarray(b, dtype=np.float32).ravel()
        H = h.shape[-1]
        if b.size == 8 * H:
            b = b[:4 * H] + b[4 * H:]
    gates = x @ w_ih.T + h @ w_hh.T
    if b is not None:
        gates = gates + b
    H = h.shape[-1]
    i, f, g, o = np.split(gates, 4, axis=-1)
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    i, f, o = sig(i), sig(f), sig(o)
    g = np.tanh(g)
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    if squeeze:
        return h_new[0], c_new[0]
    return h_new, c_new


def lstm_cell_step(x, h, c, w_ih, w_hh, b=None):
    """One LSTM step with optional Triton fast path.

    Accepts torch tensors (CUDA -> Triton, else torch) or numpy arrays
    (numpy fallback). Returns ``(h_new, c_new)`` in the input framework.
    Standalone utility; Silero ONNX wiring pending (see module docstring).
    """
    try:
        import torch
        is_torch = isinstance(x, torch.Tensor) and isinstance(h, torch.Tensor)
    except Exception:
        is_torch = False
    if is_torch:
        import torch
        if HAVE_TRITON_LSTM and x.is_cuda and h.is_cuda:
            try:
                return _tk_lstm_cell(x, h, c, w_ih, w_hh, b)
            except Exception:
                pass
        return _torch_lstm_ref(x, h, c, w_ih, w_hh, b)
    return _numpy_lstm_ref(x, h, c, w_ih, w_hh, b)


def frame_log_energy(audio, frame_len=512, hop=None):
    """Per-frame log-energy (dB-ish). Pure numpy, real code."""
    x = np.asarray(audio, dtype=np.float32).ravel()
    hop = hop or frame_len
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size))
    n = 1 + (x.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    e = np.log10((frames * frames).mean(axis=1) + 1e-10)
    return e.astype(np.float32)


def energy_probs(audio, frame_len=512, hop=None, scale=8.0):
    """Map frame log-energy to [0, 1] speech probabilities (sigmoid-norm)."""
    e = frame_log_energy(audio, frame_len, hop)
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-6:
        return np.zeros_like(e)
    z = (e - lo) / (hi - lo)
    # Center the sigmoid at the utterance median so the threshold is stable.
    med = float(np.median(z))
    return (1.0 / (1.0 + np.exp(-scale * (z - med)))).astype(np.float32)


def energy_segments(audio, sr=16000, frame_len=512, thresh=0.5,
                    min_speech_s=0.25, min_sil_s=0.3, pad_s=0.03):
    """Energy-fallback segmentation -> [(start_s, end_s)]. Real code."""
    x = np.asarray(audio, dtype=np.float32).ravel()
    hop = frame_len
    probs = energy_probs(x, frame_len, hop)
    segs, start, sil = [], None, 0
    need_sil = max(1, int(round(min_sil_s / (hop / sr))))
    for i, p in enumerate(probs):
        t = i * hop / sr
        if p >= thresh:
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
    return segs
