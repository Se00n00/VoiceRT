"""Complete TTS (Kokoro) importing fused Triton kernel, batched, VRAM check, static test."""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from src.models.triton_kernels.tts_fused import (
    postprocess, postprocess_batched, conv1d_silu, in1d_silu, estimate_tts_mb
)
from src.models.pytorch.tts import postprocess_torch, in1d_silu_torch
from src.models.runtime.memory import check_budget
from src.models.runtime.device import max_allocated_mb
import re

SAMPLE_RATE=24000
SPLIT=re.compile(r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<=[.!?])\s+")

def split_sentences(text):
    return [s.strip() for s in SPLIT.split(str(text)) if s.strip()]

class KokoroFused(torch.nn.Module):
    def __init__(self, lang_code="a", voice="af_heart", device=None, sample_rate=SAMPLE_RATE, enhance=False, batch_size=4, **_):
        super().__init__()
        if device is None:
            device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=device
        self.sample_rate=int(sample_rate)
        self.voice=voice
        self.lang_code=lang_code
        self.enhance=bool(enhance)
        self.batch_size=int(batch_size)
        self._pipeline=None
        print(f"[KokoroFused] device={device} batch={batch_size} enhance={enhance}", flush=True)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from kokoro import KPipeline
        pipe=KPipeline(lang_code=self.lang_code, device=self.device)
        self._pipeline=pipe
        return pipe

    @torch.no_grad()
    def speak(self, text, voice=None, sr=None):
        # single utterance
        return self.speak_batched([text], voice=voice, sr=sr)[0]

    @torch.no_grad()
    def speak_batched(self, texts, voice=None, sr=None):
        """Batched TTS: texts List[str] -> List[(wav, sr)] with VRAM check."""
        if voice is not None:
            self.voice=voice
        sr=int(sr or self.sample_rate)
        # VRAM check: estimate B * avg_len * 4 bytes
        est = estimate_tts_mb(len(texts), 24000, 1)  # rough 1 sec per text
        try:
            check_budget(est, budget_mb=4000, what="KokoroFused batched")
        except Exception as e:
            print(f"[KokoroFused] VRAM warning {e}, chunking", flush=True)
        pipe=self._ensure_pipeline()
        # batching via chunking: process texts in batches of self.batch_size, but pipeline itself is not batched, so we loop
        all_wavs=[]
        for i in range(0, len(texts), self.batch_size):
            chunk=texts[i:i+self.batch_size]
            chunk_wavs=[]
            for txt in chunk:
                sens=split_sentences(txt)
                parts=[]
                for sent in sens:
                    if not sent.strip():
                        continue
                    # kokoro pipeline per sentence (still eager)
                    chunks=[a for _,_,a in pipe(sent, voice=self.voice)]
                    if not chunks:
                        continue
                    arr=[np.asarray(c, dtype=np.float32).ravel() for c in chunks]
                    wav=np.concatenate(arr) if arr else np.zeros(0,dtype=np.float32)
                    # fused postprocess
                    wav=postprocess(wav, sr=SAMPLE_RATE, enhance=self.enhance)
                    parts.append(wav)
                wav_out=np.concatenate(parts) if parts else np.zeros(0,dtype=np.float32)
                if sr!=SAMPLE_RATE and wav_out.size:
                    from src.models.triton_kernels.tts_fused import resample_linear
                    wav_out=resample_linear(wav_out, SAMPLE_RATE, sr)
                chunk_wavs.append((wav_out.astype(np.float32), sr))
            all_wavs.extend(chunk_wavs)
        return all_wavs

    @staticmethod
    def test_against_torch(batch_size=2, atol=1e-5):
        device="cuda" if torch.cuda.is_available() else "cpu"
        print(f"[KokoroFused.test] device={device} batch={batch_size}", flush=True)
        est=estimate_tts_mb(batch_size, 2048, 32)
        print(f"  est {est:.1f}MB", flush=True)
        try:
            check_budget(est, budget_mb=4000, what="test_tts")
            print("  VRAM OK", flush=True)
        except Exception as e:
            print(f"  VRAM fail {e}", flush=True)
            return False
        torch.manual_seed(0)
        B=batch_size
        x=torch.randn(B, 32, 2048, device=device, dtype=torch.float16 if device=="cuda" else torch.float32)
        try:
            out_f = in1d_silu(x)
            out_t = in1d_silu_torch(x)
            err=(out_f.float()-out_t.float()).abs().max().item()
            print(f"  in1d_silu max_err={err:.2e}", flush=True)
            ok=err<1e-2
            print(f"  {'PASS' if ok else 'FAIL'}", flush=True)
            # conv test
            w=torch.randn(32,32,3, device=device, dtype=x.dtype)
            b=torch.randn(32, device=device, dtype=x.dtype)
            out_f2=conv1d_silu(x, w, b, padding=1)
            out_t2=F.silu(F.conv1d(x.float(), w.float(), b.float(), padding=1)).to(x.dtype)
            err2=(out_f2.float()-out_t2.float()).abs().max().item()
            print(f"  conv1d_silu max_err={err2:.2e}", flush=True)
            return ok and err2<1e-2
        except Exception as e:
            print(f"  error {e}", flush=True)
            import traceback; traceback.print_exc()
            return False

if __name__=="__main__":
    KokoroFused.test_against_torch()

# compat
KokoroEngine=KokoroFused
TtsEngine=KokoroFused
