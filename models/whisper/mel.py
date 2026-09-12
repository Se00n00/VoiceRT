"""80-dim log-mel frontend for Whisper (16 kHz). Pure numpy, real code.

Mirrors the HF Whisper feature extractor defaults: 25 ms window (400
samples), 10 ms hop (160 samples), 80 mel bins, log10 with a floor, no
mean normalisation (the encoder weights expect raw log-mel scaled as in
``AutoProcessor`` output up to a constant the model tolerates).
"""
import numpy as np

__all__ = [
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
    "CHUNK_LENGTH",
    "N_SAMPLES",
    "N_FRAMES",
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "log_mel_spectrogram",
    "pad_or_trim",
]

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                   fmin=0.0, fmax=8000.0):
    """Slaney-style triangular mel filterbank [n_mels, n_fft//2 + 1]."""
    n_freqs = n_fft // 2 + 1
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    bins = np.floor((n_fft + 1) * hz_edges / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        if mid > lo:
            fb[m, lo:mid] = (np.arange(lo, mid) - lo) / max(mid - lo, 1)
        if hi > mid:
            fb[m, mid:hi] = (hi - np.arange(mid, hi)) / max(hi - mid, 1)
    return fb


_FB = None


def _filters():
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    return _FB


def _stft_power(wav, n_fft=N_FFT, hop=HOP_LENGTH):
    from runtime.tensor import to_host_numpy
    x = to_host_numpy(wav)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    n = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)


def log_mel_spectrogram(wav, sr=SAMPLE_RATE, n_mels=N_MELS):
    """Raw waveform (float, any length) -> [n_mels, T] log10-mel float32."""
    from runtime.tensor import to_host_numpy
    x = to_host_numpy(wav)
    if sr != SAMPLE_RATE:
        # Cheap linear resample so callers need no extra deps.
        dur = len(x) / float(sr)
        n_out = int(round(dur * SAMPLE_RATE))
        old = np.linspace(0.0, 1.0, num=len(x))
        new = np.linspace(0.0, 1.0, num=max(n_out, 1))
        x = np.interp(new, old, x).astype(np.float32)
    power = _stft_power(x)
    mel = power @ _filters().T
    mel = np.maximum(mel, 1e-10)
    return np.log10(mel).T.astype(np.float32)


def pad_or_trim(mel, length=N_FRAMES, value=0.0):
    """[n_mels, T] -> [n_mels, length] by padding/trimming time."""
    m = np.asarray(mel, dtype=np.float32)
    if m.shape[1] > length:
        return m[:, :length]
    if m.shape[1] < length:
        pad = np.full((m.shape[0], length - m.shape[1]), value,
                      dtype=np.float32)
        return np.concatenate([m, pad], axis=1)
    return m
