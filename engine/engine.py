"""Unified voice engine: VAD + STT + LLM + TTS in one process, one VRAM pool.

Port of VOICE/voice_engine.py onto the new layout. Leg classes are imported
lazily (inside __init__/methods) so `import engine.engine` works without a
GPU, model weights, or the kokoro/transformers backends installed. Legs that
fail to load are kept as None and reported in self.missing; the method that
needs them raises RuntimeError naming the leg (omit-and-report, no stubs).
"""
import threading
import time

import numpy as np
import torch

from engine.audio import to_mono
from engine.model import load_config
from engine.session import SessionStore
from engine.streaming import SPLIT, SentenceSplitter
from runtime.device import max_allocated_mb, select_device, synchronize
from runtime.memory import check_budget
from runtime.profiler import Profiler

SYSTEM_PROMPT = "You are a voice assistant. Reply in one short spoken sentence."
STT_SR = 16000
TTS_SR = 24000
DEFAULT_VRAM_BUDGET_MB = 3800.0
DEFAULT_PER_TURN_MB = 150.0


def _pipeline_budget(config_dir=None):
    """(vram_budget_mb, per_turn_mb) from pipeline.yaml, else defaults."""
    import os

    import yaml

    budget, per_turn = DEFAULT_VRAM_BUDGET_MB, DEFAULT_PER_TURN_MB
    try:
        if config_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(os.path.dirname(here), "configs")
        path = os.path.join(config_dir, "pipeline.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        budget = float(cfg.get("vram_budget_mb", budget))
        cap = cfg.get("capacity", {}) or {}
        per_turn = float(cap.get("per_turn_mb", per_turn))
    except Exception:
        pass
    return budget, per_turn


class VoiceEngine:
    """Four legs resident; per-turn sentence-streaming LLM -> TTS."""

    def __init__(self, config_dir=None, device="cuda", max_new_tokens=48):
        self.device = select_device("cuda" if device in (None, "cuda") else device)
        self.max_new_tokens = int(max_new_tokens)
        self.profiler = Profiler()
        self.missing = []
        self.voice = "af_heart"

        cfgs = {}
        for leg in ("vad", "stt", "llm", "tts"):
            try:
                cfgs[leg] = load_config(leg, config_dir)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                self.missing.append(f"{leg} config: {exc}")
                cfgs[leg] = {}
        if cfgs["tts"].get("voice"):
            self.voice = cfgs["tts"]["voice"]
        if cfgs["llm"].get("max_new_tokens"):
            self.max_new_tokens = int(cfgs["llm"]["max_new_tokens"])
        self._whisper_id = cfgs["stt"].get("model", "openai/whisper-base")
        self._llm_id = cfgs["llm"].get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        self._vad_thresh = float(cfgs["vad"].get("threshold", 0.5))

        self.vad, self.stt, self.llm, self.tts = None, None, None, None
        self.proc, self.tok = None, None
        self.sessions = SessionStore()
        self.vram_budget_mb, self.per_turn_mb = _pipeline_budget(config_dir)
        self._load_legs()
        try:
            vram = max_allocated_mb()
        except Exception:
            vram = 0.0
        print(f"voice engine ready vram={vram:.0f}MB"
              + (f" missing=[{'; '.join(self.missing)}]" if self.missing else ""),
              flush=True)

    # -- loading (all backend imports lazy) ---------------------------------
    def _load_legs(self):
        from engine.model import load_leg

        for leg in ("vad", "stt", "llm", "tts"):
            try:
                setattr(self, leg, load_leg(leg))
            except Exception as exc:
                self.missing.append(f"{leg} leg: {exc}")
                setattr(self, leg, None)
        try:
            from transformers import AutoProcessor, AutoTokenizer

            if self.proc is None:
                self.proc = AutoProcessor.from_pretrained(self._whisper_id)
            if self.tok is None:
                self.tok = AutoTokenizer.from_pretrained(self._llm_id)
        except Exception as exc:
            self.missing.append(f"processor/tokenizer: {exc}")
        if self.tts is not None:
            try:
                self.tts.speak("warmup.")
                synchronize()
            except Exception:
                pass

    def _require(self, *names):
        lacking = [n for n in names if getattr(self, n, None) is None]
        if lacking:
            detail = "; ".join(self.missing) if self.missing else "no further detail"
            raise RuntimeError(
                f"VoiceEngine leg(s) not loaded: {', '.join(lacking)} ({detail})"
            )

    @staticmethod
    def _as_mono(audio):
        arr = np.asarray(to_mono(audio), dtype=np.float32)
        if arr.size == 0:
            raise ValueError("empty audio")
        return arr

    def _tts_chunks(self, text):
        """Run the TTS backend; return a list of float32 waveforms."""
        from runtime.tensor import to_host_numpy
        wav, _sr = self.tts.speak(text)
        return [to_host_numpy(wav)]

    # -- legs -----------------------------------------------------------------
    def vad_active(self, chunk, sr=STT_SR):
        """True when a streamed audio chunk contains speech (for endpointing).

        Cheap enough to call per chunk: runs the VAD leg on the chunk only.
        Never raises — VAD failure means "no speech" (fail-open for liveness).
        """
        try:
            self._require("vad")
            wav = np.asarray(to_mono(chunk), dtype=np.float32)
            if wav.size == 0:
                return False
            return bool(self.vad.segments(wav))
        except Exception:
            return False

    def vad_segments(self, audio, sr=STT_SR):
        """Speech segments for mono/float audio. Returns a list of spans."""
        self._require("vad")
        wav = self._as_mono(audio)
        with self.profiler.time("vad"):
            t0 = time.perf_counter()
            segs = self.vad.segments(wav)
            dt = time.perf_counter() - t0
        return {"segments": list(segs), "vad_s": dt, "dur_s": len(wav) / float(sr)}

    @torch.no_grad()
    def transcribe(self, audio, sr=STT_SR):
        """Transcribe utterance audio. Returns dict with text/rtf/ttfs/dur."""
        self._require("proc", "stt")
        wav = self._as_mono(audio)
        dur = len(wav) / float(sr)
        with self.profiler.time("stt"):
            t0 = time.perf_counter()
            feats = self.proc(wav, sampling_rate=sr, return_tensors="pt").input_features
            feats = torch.nn.functional.pad(feats, (0, 3000 - feats.shape[-1]))
            if self.device.type == "cuda":
                feats = feats.cuda()
            r = self.stt.transcribe_mel(feats, max_tokens=64)
            if isinstance(r, dict):
                ids, ttfs = r["ids"], float(r.get("ttfs", time.perf_counter() - t0))
            else:
                ids, ttfs = r, time.perf_counter() - t0
            text = self.proc.batch_decode([ids], skip_special_tokens=True)[0]
            wall = time.perf_counter() - t0
        return {
            "text": text,
            "rtf": wall / max(dur, 1e-9),
            "ttfs": ttfs,
            "dur": dur,
            "stt_s": wall,
        }

    def _chat_ids(self, text, history=None):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": text})
        return self.tok.apply_chat_template(
            msgs, return_tensors="pt",
            add_generation_prompt=True)["input_ids"][0].tolist()

    def prompt_ids(self, text, session_id=None):
        """Token ids for text, with session history prepended (if any)."""
        hist = self.sessions.history(session_id) if session_id else None
        return self._chat_ids(text, hist)

    def remember(self, session_id, user_text, assistant_text):
        """Store one turn. No-op when session_id is None (stateless)."""
        if session_id:
            self.sessions.remember_turn(session_id, user_text, assistant_text)

    def chat(self, text, max_tokens=48, stream=False, session_id=None,
             reset=False):
        """Chat reply. Returns dict with text/ttft/tps/session_id."""
        self._require("tok", "llm")
        if session_id and reset:
            self.sessions.reset(session_id)
        ids = self.prompt_ids(text, session_id)
        max_tokens = int(max_tokens)
        with self.profiler.time("llm"):
            if not stream:
                t0 = time.perf_counter()
                r = self.llm.generate(ids, max_new_tokens=max_tokens)
                if isinstance(r, dict):
                    out_ids = list(r["ids"])
                    ttft = float(r.get("ttft", time.perf_counter() - t0))
                    tps = float(r.get("decode_tps", 0.0))
                else:
                    out_ids = list(r)
                    ttft, tps = time.perf_counter() - t0, 0.0
                reply = self.tok.decode(out_ids, skip_special_tokens=True)
                self.remember(session_id, text, reply)
                return {
                    "text": reply,
                    "ttft": ttft,
                    "tps": tps,
                    "llm_s": time.perf_counter() - t0,
                    "session_id": session_id,
                }
            # Streaming path: consume the token stream, still return a dict.
            t0 = time.perf_counter()
            out_ids, first_at = [], None
            if hasattr(self.llm, "generate_stream"):
                for tok_id, _ in self.llm.generate_stream(ids, max_new_tokens=max_tokens):
                    out_ids.append(tok_id)
                    if first_at is None:
                        first_at = time.perf_counter() - t0
            else:
                out_ids, done = [], threading.Event()

                def run():
                    rr = self.llm.generate(ids, max_new_tokens=max_tokens)
                    out_ids.extend(list(rr["ids"] if isinstance(rr, dict) else rr))
                    done.set()

                th = threading.Thread(target=run)
                th.start()
                th.join()
                first_at = time.perf_counter() - t0
            wall = time.perf_counter() - t0
            reply = self.tok.decode(out_ids, skip_special_tokens=True)
            self.remember(session_id, text, reply)
            return {
                "text": reply,
                "ttft": first_at if first_at is not None else wall,
                "tps": (len(out_ids) / wall) if wall > 0 else 0.0,
                "llm_s": wall,
                "streamed": True,
                "session_id": session_id,
            }

    def speak(self, text):
        """Synthesize text. Returns dict with wav/sr/synth_s."""
        self._require("tts")
        if not text or not text.strip():
            return {"wav": np.zeros(0, dtype=np.float32), "sr": TTS_SR, "synth_s": 0.0}
        with self.profiler.time("tts"):
            t0 = time.perf_counter()
            chunks = self._tts_chunks(text.strip())
            synchronize()
            wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            dt = time.perf_counter() - t0
        return {"wav": np.asarray(wav, dtype=np.float32), "sr": TTS_SR, "synth_s": dt}

    def stream_turn(self, audio, sr=STT_SR, session_id=None, on_event=None):
        """One voice turn with LLM->TTS sentence streaming.

        Returns dict with text/reply/wav/stt_s/llm_s/ttfa_s/total_s/vram_mb.
        TTFA is measured from call entry to the first synthesized chunk.
        When session_id is given, history is prepended and the turn is
        remembered afterwards.

        When ``on_event`` is given, it is called synchronously (same thread)
        as each stage completes — STT text first, then per-token LLM pieces,
        then per-sentence TTS audio — so a streaming endpoint can forward
        partial results without waiting for the full turn:
          ("stt", {"text"}) → ("llm", {"token"})* → ("llm", {"done", "text"})
          → ("tts", {"wav", "sr"})* → return value as before.
        """
        self._require("proc", "stt", "tok", "llm", "tts")
        check_budget(self.per_turn_mb, self.vram_budget_mb, what="voice turn")
        t0 = time.perf_counter()
        st = self.transcribe(audio, sr)
        if on_event is not None:
            on_event("stt", {"text": st["text"]})
        ids = self.prompt_ids(st["text"], session_id)
        t_llm0 = time.perf_counter()
        splitter = SentenceSplitter()
        chunks, first_at, reply_ids = [], None, []

        def synth_sentence(sent):
            nonlocal first_at
            s = self.speak(sent)
            chunks.append(s["wav"])
            if first_at is None:
                first_at = time.perf_counter() - t0
            if on_event is not None:
                on_event("tts", {"wav": np.asarray(s["wav"], dtype=np.float32),
                                 "sr": TTS_SR, "sentence": sent})

        if hasattr(self.llm, "generate_stream"):
            token_iter = self.llm.generate_stream(ids, max_new_tokens=self.max_new_tokens)
            for tok_id, _ in token_iter:
                reply_ids.append(tok_id)
                piece = self.tok.decode([tok_id], skip_special_tokens=True)
                if on_event is not None:
                    on_event("llm", {"token": piece})
                for sent in splitter.push(piece):
                    synth_sentence(sent)
        else:
            # Non-streaming LLM: generate fully, then stream per sentence.
            r = self.llm.generate(ids, max_new_tokens=self.max_new_tokens)
            reply_ids = list(r["ids"] if isinstance(r, dict) else r)
            full = self.tok.decode(reply_ids, skip_special_tokens=True)
            parts = SPLIT.split(full)
            for sent in parts:
                if sent.strip():
                    synth_sentence(sent.strip())
            splitter.reset()
        tail = splitter.flush()
        if tail:
            synth_sentence(tail)
        llm_s = time.perf_counter() - t_llm0
        total = time.perf_counter() - t0
        full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        reply = self.tok.decode(reply_ids, skip_special_tokens=True)
        self.remember(session_id, st["text"], reply)
        if on_event is not None:
            on_event("llm", {"done": True, "text": reply})
        return {
            "text": st["text"],
            "reply": reply,
            "wav": np.asarray(full, dtype=np.float32),
            "sr": TTS_SR,
            "stt_s": st.get("stt_s", 0.0),
            "llm_s": llm_s,
            "ttfa_s": first_at if first_at is not None else total,
            "total_s": total,
            "vram_mb": max_allocated_mb(),
            "session_id": session_id,
        }

    def talk_turn(self, audio, sr=STT_SR, session_id=None, on_event=None):
        """Streaming-session turn: identical to stream_turn, event-first API.

        Exists so streaming callers (WS /v1/talk) don't need to know that
        ``stream_turn(..., on_event=...)`` is the same code path: pass an
        ``on_event(kind, payload)`` callback and forward each event to the
        client the moment its stage completes.
        """
        return self.stream_turn(audio, sr, session_id, on_event=on_event)
