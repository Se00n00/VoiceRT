"""Complete Whisper STT (all layers) importing fused Triton kernel.

1 fused decoder layer x6, batched, KV cache, VRAM check, static test.
"""
import glob, os, time
import numpy as np
import torch
import torch.nn.functional as F
from src.models.triton_kernels.whisper_fused import (
    whisper_fused_decoder_layer, whisper_fused_encoder_layer,
    estimate_whisper_kv_mb, layernorm
)
from src.models.pytorch.whisper import whisper_decoder_layer_torch
from src.models.runtime.memory import check_budget
from src.models.runtime.device import max_allocated_mb

D,H,DH,FF,XN = 512,8,64,2048,1500
MAXN=448; SCALE=0.125
MODEL_ID="openai/whisper-base"

def load_weights(weights_path, device="cuda:0"):
    from safetensors.torch import load_file
    if os.path.isdir(weights_path):
        files=sorted(glob.glob(os.path.join(weights_path,"*.safetensors")))
        sd={}
        for f in files:
            sd.update(load_file(f, device=device))
    else:
        sd=load_file(weights_path, device=device)
    out={}
    for k,v in sd.items():
        kk = "model."+k if k.startswith("encoder.") or k.startswith("decoder.") else k
        if not kk.startswith("model."):
            kk="model."+kk if not kk.startswith("model.") else kk
        out[kk]=v.float() if hasattr(v,"float") else v
    return out

def snapshot_path(repo_id=MODEL_ID):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"])

class WhisperFused(torch.nn.Module):
    def __init__(self, weights_path=None, device="cuda:0", model=None, batch_size=1, **_):
        super().__init__()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device="cpu"
        self.device=device
        self.model_id=model or MODEL_ID
        self.batch_size=batch_size
        if weights_path is None:
            weights_path=os.path.join(snapshot_path(self.model_id),"model.safetensors")
        self.w=load_weights(weights_path, device=device)
        print(f"[WhisperFused] device={device} batch={batch_size}", flush=True)

    def encode(self, mel, B=None):
        # mel [B,80,3000] -> memory [B,1500,512]
        if B is None:
            B=mel.shape[0]
        h = F.conv1d(mel, self.w["model.encoder.conv1.weight"], self.w["model.encoder.conv1.bias"], padding=1)
        h=F.gelu(h)
        h=F.conv1d(h, self.w["model.encoder.conv2.weight"], self.w["model.encoder.conv2.bias"], stride=2, padding=1)
        h=F.gelu(h).transpose(1,2)
        h = h + self.w["model.encoder.embed_positions.weight"][:h.shape[1]]
        for i in range(6):
            h = whisper_fused_encoder_layer(h, self.w, f"model.encoder.layers.{i}.")
        h = layernorm(h.reshape(-1,D), self.w["model.encoder.layer_norm.weight"], self.w["model.encoder.layer_norm.bias"]).reshape(h.shape)
        return h

    def cross_kv(self, memory):
        B=memory.shape[0]
        cross=[]
        for i in range(6):
            p=f"model.decoder.layers.{i}.encoder_attn."
            k=F.linear(memory, self.w[p+"k_proj.weight"], None)
            v=F.linear(memory, self.w[p+"v_proj.weight"], self.w[p+"v_proj.bias"])
            cross.append((k.view(B, XN, H, 64).transpose(1,2), v.view(B, XN, H, 64).transpose(1,2)))
        return cross

    @torch.no_grad()
    def transcribe(self, wav_batch, sr=16000, max_tokens=64, batch_size=None):
        # wav_batch: List[np array] or [B, samples]
        if batch_size is None:
            batch_size=self.batch_size
        # mel frontend batched
        from src.models.engines.whisper import log_mel_spectrogram, pad_or_trim
        mels=[]
        for wav in wav_batch:
            mel = pad_or_trim(log_mel_spectrogram(wav, sr=sr), 3000)
            mels.append(mel)
        mel_t = torch.from_numpy(np.stack(mels,0)).to(self.device)  # [B,80,3000]
        # VRAM check
        est = estimate_whisper_kv_mb(len(wav_batch), 6, H, MAXN, 64)
        check_budget(est, budget_mb=4000, what="WhisperFused transcribe")
        memory = self.encode(mel_t)
        cross = self.cross_kv(memory)
        B=len(wav_batch)
        # decoder KV cache batched [B,H,MAXN,64]
        sk = [torch.empty(B, H, MAXN, 64, device=self.device, dtype=torch.float32) for _ in range(6)]
        sv = [torch.empty(B, H, MAXN, 64, device=self.device, dtype=torch.float32) for _ in range(6)]
        Kx_batched = [c[0] for c in cross]; Vx_batched = [c[1] for c in cross]
        # greedy decode batched
        sot=50258; eot=50257
        ids = [ [sot] for _ in range(B) ]
        # use batched loop
        tokens = [ torch.tensor([sot], device=self.device) for _ in range(B) ]
        # For batched we maintain [B, D] state
        finished=[False]*B
        all_ids=[[] for _ in range(B)]
        # preallocate x [B,D] at n=0
        for n in range(max_tokens):
            # need x [B,D] from embeddings
            # need to get embeddings per batch
            emb_list=[]
            for b in range(B):
                if finished[b]:
                    # dummy
                    emb_list.append(torch.zeros(D, device=self.device))
                else:
                    tok = tokens[b][-1] if isinstance(tokens[b], list) else int(tokens[b][-1].item()) if torch.is_tensor(tokens[b]) else tokens[b]
                    # for simplicity keep ids list
                    cur_id = all_ids[b][-1] if all_ids[b] else sot
                    if n==0:
                        cur_id=sot
                    else:
                        cur_id=all_ids[b][-1] if all_ids[b] else sot
                    # actually we need nxt token
                    # we use nxt from previous step stored in tokens
                    # Simpler: use all_ids
                    if n==0:
                        cur_id=sot
                    else:
                        if finished[b]:
                            cur_id=eot
                        else:
                            cur_id=all_ids[b][-1]
                    e = F.embedding(torch.tensor([cur_id], device=self.device), self.w["model.decoder.embed_tokens.weight"]) + self.w["model.decoder.embed_positions.weight"][n]
                    emb_list.append(e[0])
            # stack
            x = torch.stack(emb_list,0)  # [B,D]
            for i in range(6):
                x = whisper_fused_decoder_layer(x, self.w, f"model.decoder.layers.{i}.", sk[i], sv[i], n, Kx_batched[i], Vx_batched[i])
            # final norm + lm head
            x = layernorm(x, self.w["model.decoder.layer_norm.weight"], self.w["model.decoder.layer_norm.bias"])
            logits = F.linear(x, self.w["model.decoder.embed_tokens.weight"])  # [B, vocab]
            nxt = logits.argmax(dim=-1)  # [B]
            for b in range(B):
                if finished[b]:
                    continue
                nid=int(nxt[b].item())
                if nid==eot:
                    finished[b]=True
                else:
                    all_ids[b].append(nid)
            if all(finished):
                break
        return {"ids": all_ids, "vram_mb": max_allocated_mb()}

    @staticmethod
    def test_against_torch(batch_size=2, atol=1e-3):
        device="cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WhisperFused.test] device={device} batch={batch_size}", flush=True)
        est=estimate_whisper_kv_mb(batch_size,6,8,448,64)
        print(f"  est KV {est:.1f}MB", flush=True)
        try:
            check_budget(est, budget_mb=4000, what="test_whisper")
            print("  VRAM OK", flush=True)
        except Exception as e:
            print(f"  VRAM fail {e}", flush=True)
            return False
        torch.manual_seed(0)
        B=batch_size
        x=torch.randn(B,D, device=device, dtype=torch.float16) * 0.5
        w={}
        def rand_w(*shape):
            return (torch.randn(*shape, device=device, dtype=torch.float32)*0.02).to(torch.float16)
        for suf in ["self_attn_layer_norm.weight","self_attn_layer_norm.bias","encoder_attn_layer_norm.weight","encoder_attn_layer_norm.bias","final_layer_norm.weight","final_layer_norm.bias"]:
            if "weight" in suf:
                w[f"model.decoder.layers.0.{suf}"]=torch.ones(D, device=device, dtype=torch.float16)
            else:
                w[f"model.decoder.layers.0.{suf}"]=torch.zeros(D, device=device, dtype=torch.float16)
        for proj in ["self_attn.q_proj.weight","self_attn.q_proj.bias","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.v_proj.bias","self_attn.out_proj.weight","self_attn.out_proj.bias","encoder_attn.q_proj.weight","encoder_attn.q_proj.bias","encoder_attn.out_proj.weight","encoder_attn.out_proj.bias","fc1.weight","fc1.bias","fc2.weight","fc2.bias"]:
            if "q_proj.weight" in proj: shape=(D,D)
            elif "k_proj" in proj: shape=(D,D)
            elif "v_proj" in proj: shape=(D,D)
            elif "out_proj" in proj: shape=(D,D)
            elif "fc1.weight" in proj: shape=(FF,D)
            elif "fc2.weight" in proj: shape=(D,FF)
            elif "fc1.bias" in proj: shape=(FF,)
            elif "fc2.bias" in proj: shape=(D,)
            else: shape=(D,)
            if "weight" in proj:
                w[f"model.decoder.layers.0.{proj}"]=rand_w(*shape) if len(shape)==2 else rand_w(shape[0])
            else:
                # bias small
                w[f"model.decoder.layers.0.{proj}"]=(torch.randn(shape[0], device=device, dtype=torch.float32)*0.01).to(torch.float16)
        sk=(torch.randn(B,H,MAXN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16) if B>1 else (torch.randn(H,MAXN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16)
        sv=(torch.randn(B,H,MAXN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16) if B>1 else (torch.randn(H,MAXN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16)
        Kx=(torch.randn(B,H,XN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16) if B>1 else (torch.randn(H,XN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16)
        Vx=(torch.randn(B,H,XN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16) if B>1 else (torch.randn(H,XN,64, device=device, dtype=torch.float32)*0.05).to(torch.float16)
        try:
            from src.models.triton_kernels.whisper_fused import whisper_fused_decoder_layer as fused
            skf=sk.clone(); svf=sv.clone()
            skt=sk.clone(); svt=sv.clone()
            out_f = fused(x.clone(), w, "model.decoder.layers.0.", skf, svf, 10, Kx, Vx)
            out_t = whisper_decoder_layer_torch(x.clone(), w, "model.decoder.layers.0.", skt, svt, 10, Kx, Vx)
            err=(out_f.float()-out_t.float()).abs().max().item()
            print(f"  max_err={err:.2e}", flush=True)
            ok=err<1e-2
            print(f"  {'PASS' if ok else 'FAIL'}", flush=True)
            return ok
        except Exception as e:
            print(f"  error {e}", flush=True)
            import traceback; traceback.print_exc()
            return False

if __name__=="__main__":
    WhisperFused.test_against_torch()
