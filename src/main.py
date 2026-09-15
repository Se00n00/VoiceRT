"""VoiceAgent: the VAD -> STT -> LLM -> TTS loop, as a LangGraph graph.

One class gathers the four model legs and compiles the turn graph.
Calling an instance streams the whole turn::

    agent = VoiceAgent()
    await agent.warm()
    async for event in agent(audio, sr=16000, session_id="..."):
        ...  # AgentEvent from vad / stt / llm / tts, then turn summary

Admission (FIFO queue + timeout) and the per-turn VRAM guard live here, so
every entry point is protected no matter the transport.
"""
import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from src.agent.events import AgentEvent  # noqa: F401  (public currency)
from src.agent.graph import build_turn_graph
from engine import SessionStore, new_session_id
from src.models.llm import LlmConfig, LlmModel
from src.models.runtime import FIFOScheduler, check_budget
from src.models.stt import SttConfig, SttModel
from src.models.tts import TtsConfig, TtsModel
from src.models.vad import VadConfig, VadModel

__all__ = ["VoiceAgentConfig", "VoiceAgent"]


@dataclass(frozen=True)
class VoiceAgentConfig:
    """Whole-agent config. Legs are dataclasses; no YAML anywhere."""

    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    max_audio_s: float = 60.0
    max_session_turns: int = 20
    session_ttl_s: float = 1800.0
    max_sessions: int = 1000
    max_inflight: int = 4
    queue_timeout_s: float = 10.0
    vram_budget_mb: float = 3800.0
    per_turn_mb: float = 150.0
    trim_pad_s: float = 0.15
    # paged LLM engine (src/inference) — shares VRAM budget, uses real QwenRunner
    llm_paged: bool = False  # set True to route LLM via paged engine (recommended on CUDA)
    llm_paged_blocks: int = 16
    llm_paged_batch_size: int = 4


class VoiceAgent:
    """VAD -> STT -> LLM -> TTS, yielding every node's stream."""

    def __init__(self, config: VoiceAgentConfig | None = None):
        self.config = config or VoiceAgentConfig()
        cfg = self.config
        # if paged LLM requested, inject paged flags into llm config
        llm_cfg = cfg.llm
        if cfg.llm_paged:
            # rebuild llm config with paged enabled (frozen dataclass)
            from dataclasses import replace
            llm_cfg = replace(llm_cfg, use_paged=True,
                              paged_blocks=cfg.llm_paged_blocks,
                              paged_batch_size=cfg.llm_paged_batch_size)
        self.vad = VadModel(cfg.vad)
        self.stt = SttModel(cfg.stt)
        self.llm = LlmModel(llm_cfg)
        self.tts = TtsModel(cfg.tts)
        self.sessions = SessionStore(
            max_turns=cfg.max_session_turns,
            max_age_s=cfg.session_ttl_s,
            max_sessions=cfg.max_sessions,
        )
        self._sched = FIFOScheduler(cfg.max_inflight)
        self.missing: list = []
        self._warmed = False
        self._graph = None  # compiled lazily: tests may swap legs first
        # paged engine handle (mirrors llm._paged_engine after warm)
        self._paged_engine = None

    @property
    def warmed(self) -> bool:
        return self._warmed

    async def warm(self) -> "VoiceAgent":
        """Build every leg; collect failures instead of raising."""
        self.missing = []
        for name in ("vad", "stt", "llm", "tts"):
            try:
                await getattr(self, name).warm()
            except Exception as exc:  # noqa: BLE001 - omit-and-report
                self.missing.append(f"{name} leg: {exc}")
        # expose paged engine if llm was built with it
        try:
            eng = None
            if hasattr(self.llm, "_paged_engine"):
                cand = getattr(self.llm, "_paged_engine")
                eng = cand() if callable(cand) else cand
                # also check inst attribute
                if eng is None and hasattr(self.llm, "_paged_engine_inst"):
                    eng = getattr(self.llm, "_paged_engine_inst")
            self._paged_engine = eng
            if self._paged_engine is None and getattr(self.config, "llm_paged", False):
                try:
                    self._paged_engine = self.llm._paged_engine()  # type: ignore
                except Exception:
                    pass
        except Exception:
            self._paged_engine = None
        self._warmed = True
        return self

    @property
    def paged_engine(self):
        # lazy fallback — only after warm to avoid eager weight fetch in tests
        if not self._warmed:
            return self._paged_engine
        if self._paged_engine is None:
            try:
                eng = self.llm._paged_engine()  # type: ignore
                self._paged_engine = eng
            except Exception:
                pass
        return self._paged_engine

    def engine_stats(self):
        """Paged engine stats if active, else None."""
        eng = self.paged_engine
        if eng is None:
            return None
        try:
            return eng.stats()  # type: ignore
        except Exception:
            return None

    # -- admission ----------------------------------------------------
    async def _admit(self):
        ticket = await asyncio.to_thread(
            self._sched.acquire, True, self.config.queue_timeout_s)
        if ticket is None:
            raise TimeoutError(
                f"server saturated ({self._sched.max_concurrency} in flight, "
                f"queue waited {self.config.queue_timeout_s:.0f}s); retry")
        return ticket

    def _compiled(self):
        """Compiled turn graph, built once from the current legs."""
        if self._graph is None:
            self._graph = build_turn_graph(
                vad=self.vad, stt=self.stt, llm=self.llm, tts=self.tts,
                sessions=self.sessions, trim_pad_s=self.config.trim_pad_s)
        return self._graph

    # -- the loop ------------------------------------------------------
    async def __call__(self, audio, sr: int = 16000, session_id: str | None = None):
        """Run one turn, yielding :class:`AgentEvent` per node stream."""
        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        if dur_s > self.config.max_audio_s:
            raise ValueError(
                f"audio {dur_s:.1f}s exceeds {self.config.max_audio_s:.0f}s cap")
        check_budget(self.config.per_turn_mb, self.config.vram_budget_mb,
                     what="voice turn")

        ticket = await self._admit()
        try:
            sid = session_id or new_session_id()
            state_in = {
                "audio": wav,
                "sr": int(sr),
                "sid": sid,
                "remember": bool(session_id),
                "history": (self.sessions.history(sid) if session_id
                            else []),
                "segments": [],
                "text": "",
                "reply_ids": [],
                "reply": "",
                "node_s": {},
                "silent": False,
                "t0": time.perf_counter(),
                "first_audio_at": None,
            }
            async for event in self._compiled().astream(
                    state_in, stream_mode="custom"):
                yield event
        finally:
            self._sched.release(ticket)
