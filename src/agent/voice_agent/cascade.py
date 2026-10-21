"""Voice cascade: VAD -> STT -> LLM -> TTS on Qwen3-0.6B.

Owns the audio legs; the LLM, chat model, sessions and scheduler stay on
the harness (``src/agent/long_running/agent.py``) and are read live.
"""

import re
import time

import numpy as np
from engine import SentenceSplitter, new_session_id
from pydantic import BaseModel, ConfigDict, Field

from src.agent.events import AgentEvent
from src.agent.long_running import cap_words, vram_mb
from src.agent.prompts.voice import VOICE_FAST_SYSTEM
from src.models.llm import LlmModel
from src.models.runtime import check_budget
from src.models.runtime.device import gpu_rev, keepalive_start
from src.models.stt import SttConfig, SttModel
from src.models.tts import TtsConfig, TtsModel
from src.models.vad import VadConfig, VadModel

__all__ = ["CascadeConfig", "CascadePipeline"]

class CascadeConfig(BaseModel):
    """Voice legs + audio caps. One class, one job.

    ..
    """

    model_config = ConfigDict(frozen=True)

    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    max_audio_s: float = 60.0
    trim_pad_s: float = 0.15
    per_turn_mb: float = 150.0
    vram_budget_mb: float = 3800.0


_TRIM_PAD_S = 0.15
MAX_SPOKEN_WORDS = 500


_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def _trim(audio, sr, segs, pad_s=_TRIM_PAD_S):
    """Cut audio to the outer speech spans plus padding on each side."""
    wav = np.asarray(audio, dtype=np.float32).ravel()
    dur = len(wav) / float(sr)
    s0 = max(0.0, float(segs[0][0]) - pad_s)
    s1 = min(dur, float(segs[-1][1]) + pad_s)
    if s1 - s0 < 0.05 or (s0 <= 0.0 and s1 >= dur):
        return audio
    return wav[int(s0 * sr) : int(s1 * sr)]


def _clauses(sentence: str) -> list[str]:
    """Split one sentence into speakable clauses on [,;:] boundaries.

    Finer than the shared sentence splitter on purpose: the first clause
    synthesizes (and starts playing) well before the sentence ends,
    which is where pause-to-first-audio time is won. The shared splitter
    stays untouched — clause chunking is a voice-turn concern only.
    """
    parts = _CLAUSE_SPLIT.split(str(sentence or ""))
    return [p.strip() for p in parts if p.strip()]


class CascadePipeline:
    """Voice legs VAD -> STT -> LLM -> TTS on Qwen3-0.6B.

    Owns the audio legs (vad/stt/tts, built here); the LLM, chat model,
    sessions and scheduler stay on the harness and are read live, so
    post-init leg swaps (model switch, eval legs) are always visible.
    One class, one job: speech in, speech out.
    ..
    """

    def __init__(self, config: CascadeConfig, harness) -> None:
        """Build the audio legs; the harness is used live, never copied.

        ..
        """
        self.config = config
        self.harness = harness
        self.vad = VadModel(config.vad)
        self.stt = SttModel(config.stt)
        self.tts = TtsModel(config.tts)

    @property
    def llm(self):
        """The worker leg for voice turns.

        ..
        """
        return self.harness.llm

    @property
    def chat_model(self):
        """Chat model for the cloud-leg fallback. Live read.

        ..
        """
        return self.harness.chat_model

    @property
    def sessions(self):
        """Session store. Live read.

        ..
        """
        return self.harness.sessions

    async def _admit(self):
        """Scheduler slot via the harness.

        ..
        """
        return await self.harness.admission.admit()

    def _release(self, ticket) -> None:
        """Release a scheduler slot via the harness.

        ..
        """
        return self.harness.admission.release(ticket)

    def _identity_sentence(self) -> str:
        """Model identity line via the harness.

        ..
        """
        return self.harness._identity_sentence()

    async def _run_locked(
        self, text, *, sid, cwd, confirm_fn, out_turns
    ):
        """Text turn via the harness loop. Yields events.

        ..
        """
        async for ev in self.harness.run_events(
            text,
            sid=sid,
            cwd=cwd,
            confirm_fn=confirm_fn,
            out_turns=out_turns,
        ):
            yield ev

    async def run_fast(self, audio, sr: int = 16000, session_id: str | None = None):
        """Latency fast path: VAD -> STT -> ONE llm.generate -> TTS.

        Same four legs, same event shapes as :meth:`run_turns`, but no
        deep-agent loop: no graph rebuild, no tool router, no multi-step
        iterations. Prompt is a small system line + capped history + the
        utterance, so prefill stays tiny and the reply ends at EOS within
        a few dozen tokens.
        """
        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        if dur_s > self.config.max_audio_s:
            raise ValueError(
                f"audio {dur_s:.1f}s exceeds {self.config.max_audio_s:.0f}s cap"
            )
        if check_budget is not None:
            check_budget(
                self.config.per_turn_mb, self.config.vram_budget_mb, what="voice turn"
            )

        ticket = await self._admit()
        stop_keep = lambda: None  # noqa: E731 - rebound below; finally-safe
        try:
            sid = session_id or new_session_id()
            remember = bool(session_id)
            eff_sid = sid if remember else None
            t0 = time.perf_counter()
            node_s: dict = {}
            stop_keep = keepalive_start()
            try:
                node_s["rev"] = float(gpu_rev())
                _rev_a = np.matmul(
                    np.zeros((128, 128), dtype=np.float32),
                    np.zeros((128, 128), dtype=np.float32),
                )
            except Exception:
                pass

            t_vad = time.perf_counter()
            try:
                segs = await self.vad.segments(wav, int(sr))
            finally:
                node_s["vad"] = node_s.get("vad", 0.0) + time.perf_counter() - t_vad
            segments = [[float(a), float(b)] for a, b in segs.segments]
            audio_wav = (
                wav
                if not segments
                else _trim(wav, sr, segments, self.config.trim_pad_s)
            )
            yield AgentEvent(
                node="vad",
                kind="segments",
                data={
                    "segments": segments,
                    "audio_dur_s": float(getattr(segs, "audio_dur_s", dur_s)),
                    "speech_s": float(getattr(segs, "speech_s", 0.0) or 0.0),
                },
            )
            if not segments:
                yield AgentEvent(
                    node="stt", kind="text", data={"text": "", "silent": True}
                )
                total = time.perf_counter() - t0
                yield AgentEvent(
                    node="turn",
                    kind="summary",
                    data={
                        "text": "",
                        "reply": "",
                        "segments": segments,
                        "node_s": dict(node_s),
                        "ttfa_s": total,
                        "total_s": total,
                        "vram_mb": vram_mb(),
                        "session_id": sid,
                    },
                )
                return

            t_stt = time.perf_counter()
            try:
                res = await self.stt.transcribe(audio_wav, int(sr))
            finally:
                node_s["stt"] = node_s.get("stt", 0.0) + time.perf_counter() - t_stt
            text = str(res.text)
            yield AgentEvent(
                node="stt",
                kind="text",
                data={
                    "text": text,
                    "rtf": float(res.rtf),
                    "ttfs": float(res.ttfs),
                    "dur_s": float(res.dur_s),
                },
            )
            if not text.strip():
                total = time.perf_counter() - t0
                yield AgentEvent(
                    node="turn",
                    kind="summary",
                    data={
                        "text": text,
                        "reply": "",
                        "segments": segments,
                        "node_s": dict(node_s),
                        "ttfa_s": total,
                        "total_s": total,
                        "vram_mb": vram_mb(),
                        "session_id": sid,
                    },
                )
                return

            t_llm = time.perf_counter()
            try:
                hist = []
                if eff_sid:
                    try:
                        for m in self.sessions.history(eff_sid)[-2:]:
                            hist.append(
                                {
                                    "role": str(m.get("role", "user")),
                                    "content": str(m.get("content", ""))[:200],
                                }
                            )
                    except Exception:
                        hist = []
                msgs = (
                    [
                        {
                            "role": "system",
                            "content": VOICE_FAST_SYSTEM
                            + " "
                            + self._identity_sentence(),
                        }
                    ]
                    + hist
                    + [{"role": "user", "content": text}]
                )
                if getattr(self, "llm", None) is not None:
                    gen = await self.llm.generate(msgs)
                    thinking, reply = LlmModel.split_thinking(str(gen.text or ""))
                    reply = str(reply or "").strip()
                else:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    _lc = [
                        SystemMessage(
                            content=VOICE_FAST_SYSTEM + " " + self._identity_sentence()
                        )
                    ]
                    for m in hist:
                        _lc.append(HumanMessage(content=str(m.get("content", ""))))
                    _lc.append(HumanMessage(content=text))
                    _out = await self.chat_model.ainvoke(_lc)
                    from src.models.provider import ChatModel

                    thinking, reply = LlmModel.split_thinking(
                        ChatModel.chat_text(getattr(_out, "content", ""))
                    )
                    reply = str(reply or "").strip()
                    gen = None
            finally:
                node_s["llm"] = node_s.get("llm", 0.0) + time.perf_counter() - t_llm
            if reply:
                try:
                    n_ids = (
                        len(getattr(gen, "output_ids", ()) or ())
                        if gen is not None
                        else 0
                    )
                except Exception:
                    n_ids = 0
                yield AgentEvent(
                    node="llm", kind="done", data={"text": reply, "ids": n_ids}
                )
            if eff_sid and (text or reply):
                try:
                    self.sessions.remember_turn(eff_sid, str(text), reply)
                except Exception:
                    pass

            first_audio_at = None
            reply = cap_words(reply)
            if (reply or "").strip():
                t_tts = time.perf_counter()
                try:
                    out = await self.tts.speak(reply)
                finally:
                    node_s["tts"] = node_s.get("tts", 0.0) + time.perf_counter() - t_tts
                first_audio_at = time.perf_counter() - t0
                yield AgentEvent(
                    node="tts",
                    kind="audio",
                    data={
                        "wav": np.asarray(out.wav, dtype=np.float32),
                        "sr": int(out.sample_rate),
                        "sentence": str(out.sentence or reply),
                        "synth_s": float(out.synth_s),
                    },
                )

            total = time.perf_counter() - t0
            yield AgentEvent(
                node="turn",
                kind="summary",
                data={
                    "text": text,
                    "reply": reply,
                    "segments": segments,
                    "node_s": dict(node_s),
                    "ttfa_s": first_audio_at if first_audio_at is not None else total,
                    "total_s": total,
                    "vram_mb": vram_mb(),
                    "session_id": sid,
                },
            )
        finally:
            try:
                stop_keep()
            except Exception:
                pass
            self._release(ticket)

    async def run_turns(self, audio, sr: int = 16000, session_id: str | None = None):
        """Run one voice turn, yielding vad/stt/llm/tts/term events + summary."""
        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        if dur_s > self.config.max_audio_s:
            raise ValueError(
                f"audio {dur_s:.1f}s exceeds {self.config.max_audio_s:.0f}s cap"
            )
        if check_budget is not None:
            check_budget(
                self.config.per_turn_mb, self.config.vram_budget_mb, what="voice turn"
            )

        ticket = await self._admit()
        try:
            sid = session_id or new_session_id()
            remember = bool(session_id)
            t0 = time.perf_counter()
            node_s: dict = {}

            t_vad = time.perf_counter()
            try:
                segs = await self.vad.segments(wav, int(sr))
            finally:
                node_s["vad"] = node_s.get("vad", 0.0) + time.perf_counter() - t_vad
            segments = [[float(a), float(b)] for a, b in segs.segments]
            audio_wav = (
                wav
                if not segments
                else _trim(wav, sr, segments, self.config.trim_pad_s)
            )
            yield AgentEvent(
                node="vad",
                kind="segments",
                data={
                    "segments": segments,
                    "audio_dur_s": float(getattr(segs, "audio_dur_s", dur_s)),
                    "speech_s": float(getattr(segs, "speech_s", 0.0) or 0.0),
                },
            )
            if not segments:
                yield AgentEvent(
                    node="stt", kind="text", data={"text": "", "silent": True}
                )
                total = time.perf_counter() - t0
                yield AgentEvent(
                    node="turn",
                    kind="summary",
                    data={
                        "text": "",
                        "reply": "",
                        "segments": segments,
                        "node_s": dict(node_s),
                        "ttfa_s": total,
                        "total_s": total,
                        "vram_mb": vram_mb(),
                        "session_id": sid,
                    },
                )
                return

            t_stt = time.perf_counter()
            try:
                res = await self.stt.transcribe(audio_wav, int(sr))
            finally:
                node_s["stt"] = node_s.get("stt", 0.0) + time.perf_counter() - t_stt
            text = str(res.text)
            yield AgentEvent(
                node="stt",
                kind="text",
                data={
                    "text": text,
                    "rtf": float(res.rtf),
                    "ttfs": float(res.ttfs),
                    "dur_s": float(res.dur_s),
                },
            )
            if not text.strip():
                total = time.perf_counter() - t0
                yield AgentEvent(
                    node="turn",
                    kind="summary",
                    data={
                        "text": text,
                        "reply": "",
                        "segments": segments,
                        "node_s": dict(node_s),
                        "ttfa_s": total,
                        "total_s": total,
                        "vram_mb": vram_mb(),
                        "session_id": sid,
                    },
                )
                return

            t_llm = time.perf_counter()
            first_audio_at = None
            first_piece = True
            turns: list = []
            eff_sid = sid if remember else None
            async for ev in self._run_locked(
                text,
                sid=eff_sid,
                cwd=".",
                confirm_fn=None,
                out_turns=turns,
            ):
                if ev.kind == "token":
                    yield AgentEvent(
                        node="llm",
                        kind="token",
                        data={
                            "token": str(ev.data.get("piece", "") or ""),
                            "piece": str(ev.data.get("piece", "") or ""),
                            "first": first_piece,
                        },
                    )
                    first_piece = False
                elif ev.kind == "thinking":
                    yield AgentEvent(
                        node="llm",
                        kind="thinking",
                        data={"text": ev.data.get("text", "")},
                    )
                elif ev.kind in (
                    "action",
                    "observation",
                    "confirm",
                    "deny",
                    "error",
                    "queued",
                    "route",
                ):
                    yield ev
            node_s["llm"] = node_s.get("llm", 0.0) + time.perf_counter() - t_llm

            for _t, reply, _th in turns:
                reply = cap_words(reply)
                if not (reply or "").strip():
                    continue
                splitter = SentenceSplitter()
                pending: list[str] = []

                async def _speak_sentence(sent):
                    """Speak one sentence and emit its audio event."""
                    nonlocal first_audio_at
                    t_tts = time.perf_counter()
                    try:
                        out = await self.tts.speak(sent)
                    finally:
                        node_s["tts"] = (
                            node_s.get("tts", 0.0) + time.perf_counter() - t_tts
                        )
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter() - t0
                    yield AgentEvent(
                        node="tts",
                        kind="audio",
                        data={
                            "wav": np.asarray(out.wav, dtype=np.float32),
                            "sr": int(out.sample_rate),
                            "sentence": str(out.sentence or sent),
                            "synth_s": float(out.synth_s),
                        },
                    )

                async def _feed(sentences, speak):
                    """Speak each non-blank sentence through the synth."""
                    for sent in sentences:
                        if (sent or "").strip():
                            async for ev in speak(sent):
                                yield ev

                for sent in splitter.push(reply):
                    pending.append(sent)
                tail = splitter.flush()
                if tail:
                    pending.append(tail)
                # Clause fan-out: each sentence speaks in [,;:] chunks so
                # the first audio starts after the first clause, not the
                # first sentence. Client-side playback already overlaps
                # synth of later chunks with play of earlier ones.
                clauses = [c for sent in pending for c in _clauses(sent)]
                async for audio_ev in _feed(
                    [c for c in clauses if (c or "").strip()], _speak_sentence
                ):
                    yield audio_ev
                yield AgentEvent(node="llm", kind="done", data={"text": reply})

            total = time.perf_counter() - t0
            last_reply = turns[-1][1] if turns else ""
            yield AgentEvent(
                node="turn",
                kind="summary",
                data={
                    "text": text,
                    "reply": last_reply,
                    "segments": segments,
                    "node_s": dict(node_s),
                    "ttfa_s": first_audio_at if first_audio_at is not None else total,
                    "total_s": total,
                    "vram_mb": vram_mb(),
                    "session_id": sid,
                },
            )
        finally:
            self._release(ticket)


