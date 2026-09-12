"""Serial voice pipeline: wav -> STT -> LLM -> TTS wav (port of VOICE/pipeline.py).

Adapted to the new layout: device/profiler/audio/model helpers come from
runtime.* / engine.*, leg classes load lazily from models.* via
engine.model.load_leg with an HF fallback. Timings + return contract match
the prototype: run(wav_path) -> (wav_24k, timings_dict).
"""
import os
import time

import numpy as np
import torch

from engine.audio import load_wav
from engine.model import load_config
from runtime.device import max_allocated_mb, select_device, synchronize
from runtime.profiler import Profiler

SYSTEM_PROMPT = "You are a voice assistant. Reply in one short spoken sentence."
TARGET_STT_SR = 16000
TARGET_TTS_SR = 24000


class VoicePipe:
    """All legs resident; serial handoff STT -> LLM -> TTS."""

    def __init__(self, config_dir=None, device=None, voice="af_heart", max_new_tokens=48):
        self.device = select_device("cuda" if device in (None, "cuda") else device)
        self.voice = voice
        self.max_new_tokens = int(max_new_tokens)
        self.profiler = Profiler()
        self.missing = []

        stt_cfg, llm_cfg, tts_cfg = {}, {}, {}
        for leg, slot in (("stt", "stt_cfg"), ("llm", "llm_cfg"), ("tts", "tts_cfg")):
            try:
                cfg = load_config(leg, config_dir)
            except (FileNotFoundError, ValueError) as exc:
                self.missing.append(f"{leg} config: {exc}")
                cfg = {}
            if slot == "stt_cfg":
                stt_cfg = cfg
            elif slot == "llm_cfg":
                llm_cfg = cfg
            else:
                tts_cfg = cfg
        if tts_cfg.get("voice"):
            self.voice = tts_cfg["voice"]
        if llm_cfg.get("max_new_tokens"):
            self.max_new_tokens = int(llm_cfg["max_new_tokens"])
        self.whisper_id = stt_cfg.get("model", "openai/whisper-base")
        self.llm_id = llm_cfg.get("model", "Qwen/Qwen2.5-0.5B-Instruct")

        self.proc, self.stt, self.tok, self.llm, self.tts = self._load_legs()

    def _load_legs(self):
        """Lazy backend imports (transformers/kokoro stay out of module top)."""
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        from engine.model import load_leg

        try:
            stt = load_leg("stt")
        except Exception as exc:
            self.missing.append(f"stt leg: {exc}")
            stt = None
        try:
            llm_leg = load_leg("llm")
        except Exception as exc:
            self.missing.append(f"llm leg: {exc}")
            llm_leg = None
        try:
            tts = load_leg("tts")
        except Exception as exc:
            self.missing.append(f"tts leg: {exc}")
            tts = None

        proc = AutoProcessor.from_pretrained(self.whisper_id)
        tok = AutoTokenizer.from_pretrained(self.llm_id)
        if llm_leg is None:
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_id,
                torch_dtype=torch.float16,
                device_map="cuda:0" if self.device.type == "cuda" else "cpu",
            ).eval()
        else:
            llm = llm_leg
        if tts is None:
            from kokoro import KPipeline
            tts = KPipeline(lang_code="a", device="cuda" if self.device.type == "cuda" else "cpu")
        missing_stt_note = [m for m in self.missing if m.startswith("stt leg")]
        if stt is None and not missing_stt_note:
            self.missing.append("stt leg: no models.whisper backend; using HF processor only")
        return proc, stt, tok, llm, tts

    def _stt_text(self, audio, sr):
        feats = self.proc(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = torch.nn.functional.pad(feats, (0, 3000 - feats.shape[-1]))
        feats = feats.to(self.device if self.device.type == "cuda" else "cpu")
        if self.stt is not None and hasattr(self.stt, "transcribe"):
            r = self.stt.transcribe(feats)
            ids = r["ids"] if isinstance(r, dict) else r
            return self.proc.batch_decode([ids], skip_special_tokens=True)[0]
        # Fallback: HF whisper model when no custom STT leg is installed.
        from transformers import AutoModelForSpeechSeq2Seq
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.whisper_id).to(
            self.device if self.device.type == "cuda" else "cpu"
        ).eval()
        with torch.no_grad():
            out = model.generate(feats)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]

    @torch.no_grad()
    def run(self, wav_path):
        """Run one file end-to-end. Returns (wav_24k_float32, timings_dict)."""
        t = {}
        audio, sr = load_wav(wav_path, target_sr=TARGET_STT_SR)
        t["audio_dur"] = len(audio) / sr
        t0 = time.perf_counter()
        with self.profiler.time("stt"):
            t_stt0 = time.perf_counter()
            text = self._stt_text(audio, sr)
            t["stt"] = time.perf_counter() - t_stt0
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        ids = self.tok.apply_chat_template(msgs, return_tensors="pt",
                                           add_generation_prompt=True)
        cuda = self.device.type == "cuda"
        nid = ids["input_ids"].to("cuda:0" if cuda else "cpu")
        with self.profiler.time("llm"):
            t1 = time.perf_counter()
            if hasattr(self.llm, "generate") and not hasattr(self.llm, "config"):
                r = self.llm.generate(nid[0].tolist(), max_new_tokens=self.max_new_tokens)
                if isinstance(r, dict) and "ids" in r:
                    reply = self.tok.decode(r["ids"], skip_special_tokens=True)
                else:
                    reply = self.tok.decode(list(r), skip_special_tokens=True)
            else:
                out = self.llm.generate(nid, max_new_tokens=self.max_new_tokens, do_sample=False)
                synchronize()
                reply = self.tok.decode(out[0][nid.shape[1]:], skip_special_tokens=True)
            t["llm"] = time.perf_counter() - t1
        with self.profiler.time("tts"):
            t2 = time.perf_counter()
            chunks = [a for _, _, a in self.tts(reply, voice=self.voice)]
            synchronize()
            t["tts"] = time.perf_counter() - t2
        wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        t["total"] = time.perf_counter() - t0
        t["text"], t["reply"] = text, reply
        t["vram"] = max_allocated_mb()
        return np.asarray(wav, dtype=np.float32), t


def main():
    import json

    from engine.audio import save_wav

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(here), "samples_out")
    os.makedirs(out_dir, exist_ok=True)
    pipe = VoicePipe()
    if pipe.missing:
        print("notes: " + " | ".join(pipe.missing), flush=True)
    with open("STT/samples/samples.json") as f:
        manifest = json.load(f)
    for m in manifest:
        wav, t = pipe.run(m["wav"])
        name = os.path.basename(m["wav"]).replace(".wav", "")
        save_wav(os.path.join(out_dir, f"pipe_{name}.wav"), wav, TARGET_TTS_SR)
        print(f"{name}: audio={t['audio_dur']:.1f}s "
              f"stt={t['stt']*1000:.0f}ms llm={t['llm']*1000:.0f}ms "
              f"tts={t['tts']*1000:.0f}ms E2E={t['total']*1000:.0f}ms "
              f"vram={t['vram']:.0f}MB", flush=True)
        print(f"  user: {t['text'][:70]!r}", flush=True)
        print(f"  asst: {t['reply'][:70]!r}", flush=True)
    print("outputs in", out_dir)


if __name__ == "__main__":
    main()
