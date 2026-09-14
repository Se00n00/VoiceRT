"""Whisper-base leg in one file: weights + mel + encoder + decoder + engine.

Single model class :class:`WhisperEngine`. All Triton kernels come from
:mod:`src.models.triton_kernels.whisper` (layernorm, row_softmax, decode_attn,
batched_decode_attn, fused_qkv) with exact torch fallbacks. Every exported
kernel is in the hot path:

- ``layernorm``: every encoder/decoder layer + final norms.
- ``row_softmax``: inside ``enc_self_attention`` (CUDA path).
- ``decode_attn``: decoder self-attention steps.
- ``batched_decode_attn``: decoder cross-attention steps (B=1 slice).
- ``fused_qkv``: decoder self-attention QKV projection.
"""
import glob
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.models.triton_kernels.whisper import (
    HAVE_TRITON_KERNELS,
    batched_decode_attn,
    decode_attn,
    decode_cross_attention_step,
    decode_self_attention_step,
    enc_self_attention,
    fused_qkv,
    layernorm,
    row_softmax,
)

__all__ = [
    "D", "H", "DH", "FF", "XN", "MAXN", "SCALE",
    "SOT", "EOT", "MODEL_ID", "N_FRAMES", "SAMPLE_RATE",
    "WhisperEngine", "WhisperTriton", "WhisperModel",
    "HAVE_TRITON_KERNELS",
]

D, H, DH, FF, XN = 512, 8, 64, 2048, 1500
MAXN = 448
SCALE = 0.125

MODEL_ID = "openai/whisper-base"
SOT = 50258  # <|startoftranscript|> for whisper-base
EOT = 50257  # <|endoftext|>

REQUIRED_PREFIXES = (
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
    "model.encoder.embed_positions.weight",
    "model.encoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.embed_tokens.weight",
    "model.decoder.embed_positions.weight",
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layer_norm.weight",
)

# -- mel frontend constants -------------------------------------------
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000

N_ENC_LAYERS = 6
N_DEC_LAYERS = 6

_SPECIAL = re.compile(r"<\|[^|]*\|>")


# -- weights ------------------------------------------------------------
def snapshot_path(repo_id=MODEL_ID, local_dir=None):
    """Local HF snapshot dir (lazy huggingface_hub), downloading if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"],
                             local_dir=local_dir)


def map_key(key):
    """Canonicalise one state-dict key to the ``model.*`` layout."""
    if key.startswith("model."):
        return key
    if key.startswith("encoder.") or key.startswith("decoder."):
        return "model." + key
    return key


def load_weights(weights_path, device="cuda:0"):
    """Load ``.safetensors`` file or directory -> {name: float tensor}."""
    from safetensors.torch import load_file
    if os.path.isdir(weights_path):
        files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                "no .safetensors under %s" % weights_path)
        sd = {}
        for f in files:
            sd.update(load_file(f, device=device))
    else:
        sd = load_file(weights_path, device=device)
    out = {}
    for k, v in sd.items():
        out[map_key(k)] = v.float() if hasattr(v, "float") else v
    return out


def check_coverage(weights):
    """Verify required prefixes exist. Returns (ok, missing_list)."""
    keys = set(weights.keys())
    missing = [p for p in REQUIRED_PREFIXES
               if not any(k == p or k.startswith(p) for k in keys)]
    tied_ok = ("model.decoder.embed_tokens.weight" in keys)  # LM head tied
    return (not missing and tied_ok, missing)


# -- mel frontend ---------------------------------------------------------
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
    from src.models.runtime.tensor import to_host_numpy
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
    from src.models.runtime.tensor import to_host_numpy
    x = to_host_numpy(wav)
    if sr != SAMPLE_RATE:
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


# -- tokenizer ------------------------------------------------------------
def _get_processor(model_id=MODEL_ID):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


def byte_fallback_decode(ids):
    """Minimal GPT-2 byte-level decode for offline use (real code)."""
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


# -- encoder --------------------------------------------------------------
def _ln(x, w, b):
    # reshape may return a non-contiguous VIEW; kernels need dense rows
    return layernorm(x.reshape(-1, D).contiguous(), w, b).reshape(x.shape)


def encoder_layer(x, w, i):
    """One encoder layer with residual adds. x [B, T, D] -> [B, T, D]."""
    p = "model.encoder.layers.%d." % i
    h = _ln(x, w[p + "self_attn_layer_norm.weight"],
            w[p + "self_attn_layer_norm.bias"])
    x = x + enc_self_attention(h, w, p)
    h = _ln(x, w[p + "final_layer_norm.weight"],
            w[p + "final_layer_norm.bias"])
    m = F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])
    m = F.gelu(m)
    return x + F.linear(m, w[p + "fc2.weight"],
                       w[p + "fc2.bias"]).view(x.shape[0], x.shape[1], D)


def encode_mel(mel, w, n_layers=N_ENC_LAYERS):
    """mel [B,80,3000] -> memory [B,1500,512]."""
    h = F.conv1d(mel, w["model.encoder.conv1.weight"],
                 w["model.encoder.conv1.bias"], padding=1)
    h = F.gelu(h)
    h = F.conv1d(h, w["model.encoder.conv2.weight"],
                 w["model.encoder.conv2.bias"], stride=2, padding=1)
    h = F.gelu(h).transpose(1, 2)
    h = h + w["model.encoder.embed_positions.weight"][:h.shape[1]]
    for i in range(n_layers):
        h = encoder_layer(h, w, i)
    h = layernorm(h.reshape(-1, D), w["model.encoder.layer_norm.weight"],
                  w["model.encoder.layer_norm.bias"]).reshape(h.shape)
    return h


def cross_kv(memory, w, n_layers=N_ENC_LAYERS):
    """memory [B,1500,512] -> per-layer (K, V) cross-attention caches."""
    B = memory.shape[0]
    cross = []
    for i in range(n_layers):
        p = "model.decoder.layers.%d.encoder_attn." % i
        k = F.linear(memory, w[p + "k_proj.weight"], None)
        v = F.linear(memory, w[p + "v_proj.weight"], w[p + "v_proj.bias"])
        cross.append((
            k.view(B, XN, H, 64).transpose(1, 2),
            v.view(B, XN, H, 64).transpose(1, 2),
        ))
    # Squeeze the batch dim for the single-utterance path.
    if B == 1:
        cross = [(k[0].reshape(H, XN, 64), v[0].reshape(H, XN, 64))
                 for k, v in cross]
    return cross


# -- decoder --------------------------------------------------------------
def decoder_step(x, w, i, sk, sv, n, Kx, Vx):
    """One decoder layer at position ``n``. x [D] -> [D]."""
    p = "model.decoder.layers.%d." % i
    h = _ln(x, w[p + "self_attn_layer_norm.weight"],
            w[p + "self_attn_layer_norm.bias"])
    x = x + decode_self_attention_step(h, w, p, sk, sv, n)
    h = _ln(x, w[p + "encoder_attn_layer_norm.weight"],
            w[p + "encoder_attn_layer_norm.bias"])
    x = x + decode_cross_attention_step(h, w, p, Kx, Vx)
    h = _ln(x, w[p + "final_layer_norm.weight"],
            w[p + "final_layer_norm.bias"])
    m = F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])
    return x + F.linear(F.gelu(m), w[p + "fc2.weight"], w[p + "fc2.bias"])


@torch.no_grad()
def greedy_decode(w, cross, device, max_tokens=MAXN, sot=SOT, eot=EOT,
                  n_layers=N_DEC_LAYERS):
    """Greedy loop over cached cross KV -> token id list."""
    sk = [torch.empty(8, MAXN, 64, device=device) for _ in range(n_layers)]
    sv = [torch.empty(8, MAXN, 64, device=device) for _ in range(n_layers)]
    tok = torch.tensor([sot], device=device)
    ids = []
    for n in range(min(max_tokens, MAXN)):
        e = F.embedding(tok, w["model.decoder.embed_tokens.weight"])
        e = e + w["model.decoder.embed_positions.weight"][n]
        x = e.reshape(D)
        for i in range(n_layers):
            x = decoder_step(x, w, i, sk[i], sv[i], n, *cross[i])
        x = layernorm(x.reshape(1, D), w["model.decoder.layer_norm.weight"],
                      w["model.decoder.layer_norm.bias"]).reshape(D)
        # LM head is tied to decoder embeddings (no separate tensor/bias)
        nxt = F.linear(
            x, w["model.decoder.embed_tokens.weight"]).argmax().item()
        ids.append(nxt)
        if nxt == eot:
            break
        tok = torch.tensor([nxt], device=device)
    return ids


# -- engine ---------------------------------------------------------------
class WhisperEngine:
    """Working whisper-base inference engine (Triton kernels + cuBLAS).

    Takes explicit kwargs (``model``, ``language``, ``max_new_tokens``,
    ...).
    """

    def __init__(self, weights_path=None, device="cuda:0", model=None,
                 language="en", sample_rate=SAMPLE_RATE,
                 max_new_tokens=64, kernels=None, **_ignored):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_id = model or MODEL_ID
        self.language = language
        self.sample_rate = int(sample_rate)
        self.default_max_tokens = int(max_new_tokens)
        if weights_path is None:
            weights_path = os.path.join(snapshot_path(self.model_id),
                                        "model.safetensors")
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
        """Raw waveform -> text dict. Runs mel frontend + decode + detokenize."""
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


WhisperTriton = WhisperEngine
WhisperModel = WhisperEngine
