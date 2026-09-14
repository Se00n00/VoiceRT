"""WAV load/save/resample via soundfile + numpy (linear-interp resampling)."""
import os

import numpy as np
import soundfile as sf


def to_mono(audio):
    """Average multi-channel audio to mono; pass through 1-D input."""
    from src.models.runtime.tensor import to_host_numpy
    audio = to_host_numpy(audio, dtype=np.float64)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"to_mono expects 1-D or 2-D audio, got shape {audio.shape}")


def resample(audio, src_sr, dst_sr):
    """Resample a 1-D float array with linear interpolation (real code, no scipy)."""
    audio = np.asarray(to_mono(audio), dtype=np.float64)
    src_sr, dst_sr = int(src_sr), int(dst_sr)
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"sample rates must be > 0, got {src_sr} -> {dst_sr}")
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.shape[0] / src_sr
    n_out = max(1, int(round(duration * dst_sr)))
    src_t = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def load_wav(path, target_sr=None, mono=True, dtype=np.float32):
    """Load a wav file; optionally resample and force mono.

    Returns (audio_1d_float32, sample_rate).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"wav not found: {path}")
    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio)
    if mono and audio.ndim == 2:
        audio = to_mono(audio)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if target_sr is not None and int(target_sr) != int(sr):
        if audio.ndim == 2:
            audio = np.stack(
                [resample(audio[:, c], sr, target_sr) for c in range(audio.shape[1])],
                axis=1,
            )
            if mono:
                audio = to_mono(audio)
        else:
            audio = resample(audio, sr, target_sr)
        sr = int(target_sr)
    return np.asarray(audio, dtype=dtype), int(sr)


def save_wav(path, audio, sr):
    """Write mono/stereo float audio to wav, creating parent dirs."""
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"sample rate must be > 0, got {sr}")
    audio = np.asarray(audio)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr)
    return path


def normalize(audio, peak=0.99):
    """Scale float audio so |max| == peak (silence passes through)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    amp = float(np.max(np.abs(audio)))
    if amp <= 1e-9:
        return audio
    return (audio * (peak / amp)).astype(np.float32)


def duration_s(audio, sr):
    """Duration of an audio array in seconds."""
    audio = np.asarray(audio)
    n = audio.shape[0] if audio.ndim else 0
    return float(n) / float(sr) if sr else 0.0
