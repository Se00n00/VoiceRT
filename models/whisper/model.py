"""Whisper engine: mel frontend call, encoder, decode loop, wav -> text.

Ported from ``STT/triton/engine.py`` (``WhisperTriton``). Accepts either a
precomputed mel ``[1,80,3000]`` tensor (original call path) or a raw 16 kHz
waveform via :meth:`WhisperEngine.transcribe`.
"""
import time

import torch

from models.whisper.decoder import greedy_decode
from models.whisper.encoder import cross_kv, encode_mel

__all__ = ["D", "H", "DH", "FF", "XN", "MAXN", "SCALE", "WhisperEngine"]

D, H, DH, FF, XN = 512, 8, 64, 2048, 1500
MAXN = 448
SCALE = 0.125


class WhisperEngine:
    """Working whisper-base inference engine (Triton kernels + cuBLAS)."""

    def __init__(self, weights_path=None, device="cuda:0"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        if weights_path is None:
            from models.whisper.weights import snapshot_path
            import os
            weights_path = os.path.join(snapshot_path(), "model.safetensors")
        from models.whisper.weights import load_weights
        self.w = load_weights(weights_path, device=device)
        if device.startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    # -- encoder ------------------------------------------------------
    def encode(self, mel):
        """mel [1,80,3000] -> per-layer cross KV (memory path in engine)."""
        memory = encode_mel(mel, self.w)
        return cross_kv(memory, self.w)

    # -- decode loop --------------------------------------------------
    @torch.no_grad()
    def transcribe_mel(self, mel, max_tokens=64):
        """mel [1,80,3000] tensor -> dict(ids, ttfs, total, vram_mb)."""
        t0 = time.perf_counter()
        mel = mel.to(self.device)
        cross = self.encode(mel)
        first_t = time.perf_counter()
        ids = greedy_decode(self.w, cross, self.device,
                            max_tokens=max_tokens)
        total = time.perf_counter() - t0
        vram = 0.0
        if self.device.startswith("cuda"):
            try:
                vram = torch.cuda.max_memory_allocated() / 1024 ** 2
            except Exception:
                vram = 0.0
        return {"ids": ids, "ttfs": first_t - t0, "total": total,
                "vram_mb": vram}

    # -- full wav -> text ---------------------------------------------
    @torch.no_grad()
    def transcribe(self, wav, sr=16000, max_tokens=64, model_id=None):
        """Raw waveform -> text dict. Runs mel frontend + decode + detokenize.

        ``transformers`` is imported lazily so this module imports cleanly
        without it (then the bundled byte-fallback decoder is used).
        """
        import numpy as np

        from models.whisper.mel import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from models.whisper.tokenizer import clean_text, decode_ids
        mel = pad_or_trim(log_mel_spectrogram(wav, sr=sr), N_FRAMES)
        mel_t = torch.from_numpy(np.ascontiguousarray(mel)).unsqueeze(0).to(
            self.device)
        r = self.transcribe_mel(mel_t, max_tokens=max_tokens)
        text = decode_ids(r["ids"]) if model_id is None else decode_ids(
            r["ids"], model_id=model_id)
        w = wav.detach().cpu() if hasattr(wav, "cpu") else wav
        dur = len(np.asarray(w).ravel()) / float(sr)
        return {"text": clean_text(text), "ids": r["ids"],
                "rtf": r["total"] / max(dur, 1e-9),
                "ttfs": r["ttfs"], "dur": dur, "vram_mb": r["vram_mb"]}
