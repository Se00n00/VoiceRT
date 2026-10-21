"""VoiceAgent: ONE autonomous agent for voice, terminal, and everything else.

Device placement:

- LLM  Qwen3-0.6B fused Triton (``backend="qwen"``) or Bonsai 27B
  llama.cpp sidecar (``backend="bonsai"``) for agentic work.
- VAD  Silero ONNX (CPU, RTF ~0.01).
- STT  Whisper-base pure-torch (GPU when CUDA is up, CPU fallback otherwise).
- TTS  Kokoro-82M + torch post (GPU when CUDA is up, CPU fallback otherwise).

Pipeline: speech -> VAD spans -> STT text -> deep-agent turn (tools +
JSON session memory) -> TTS. Text turns go through :meth:`run_text`
(text clients, CLI, evals); voice turns through :meth:`__call__`
(``server.py`` ``/talk``). If a turn is already running for a session,
new input is queued and injected as follow-up message(s) once the
active turn drains.
"""

from pydantic import BaseModel, ConfigDict, Field

from src.agent.long_running import LongRunningAgent, LongRunningConfig
from src.agent.voice_agent.cascade import CascadeConfig, CascadePipeline

__all__ = ["VoiceAgentConfig", "VoiceAgent"]


class VoiceAgentConfig(BaseModel):
    """Shell config: cascade + harness sub-configs.

    Flat kwargs are gone: pass cascade=/harness= configs.
    ..
    """

    model_config = ConfigDict(frozen=True)

    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    harness: LongRunningConfig = Field(default_factory=LongRunningConfig)
    fast_voice: bool = False


class VoiceAgent(LongRunningAgent):
    """Text + voice entry points.

    Behavioral shell over the harness + cascade: every attribute and
    method external callers use keeps working unchanged.
    ..
    """

    def __init__(self, config: VoiceAgentConfig | None = None) -> None:
        """Build the harness, then the cascade on it. Warms nothing.

        ..
        """
        cfg = config or VoiceAgentConfig()
        super().__init__(cfg.harness)
        self.cascade = CascadePipeline(cfg.cascade, self)
        self.tts = self.cascade.tts
        self.fast_voice = bool(cfg.fast_voice)

    def __call__(self, audio_or_text, sr: int = 16000, session_id: str | None = None):
        """Text or voice dispatch (both transports share one object).

        - ``await agent("compute it", session_id="t")`` — plain text turn,
          returns the final reply string (CLI-style / unit tests).
        - ``async for ev in agent(wav, sr, sid)`` — voice turn, yields
          vad/stt/llm/tts/term events + summary (``server.py`` ``/talk``).
        """
        if isinstance(audio_or_text, str):
            return self._text_reply(str(audio_or_text), session_id=session_id)
        if bool(self.fast_voice):
            return self._voice_fast(audio_or_text, sr=sr, session_id=session_id)
        return self._voice_turns(audio_or_text, sr=sr, session_id=session_id)

    async def _text_reply(self, text: str, session_id: str | None = None) -> str:
        """One text turn, reply string only (no audio synthesis)."""
        reply = ""
        async for ev in self.run_text(
            str(text), session_id=session_id, speak=False
        ):
            if ev.kind == "chat":
                reply = str((ev.data or {}).get("reply", "") or "")
        return reply

    # ---- voice agent (audio legs: fast path, full voice turns) ----

    @property
    def warmed(self) -> bool:
        """Whether warm() has completed at least once."""
        return self._warmed

    async def warm(self) -> "VoiceAgent":
        """Build every leg; collect failures instead of raising."""
        self.missing = []
        self.worker.unavailable = None
        for name in ("vad", "stt", "tts", "llm"):
            if name == "tts" and not self.config.speak_text_turns:
                self.missing.append("tts leg: skipped (text-only)")
                continue
            if name == "llm" and not self.worker.eager:
                continue
            leg = getattr(self.cascade, name, None) or getattr(self, name, None)
            if leg is None:
                continue
            try:
                await leg.warm()
                if name == "llm":
                    self.worker.warmed = True
            except Exception as exc:  # noqa: BLE001 - omit-and-report
                self.missing.append(f"{name} leg: {exc}")
        if (
            getattr(self, "llm", None) is None
            and getattr(self, "chat_model", None) is not None
        ):
            try:
                self.worker.warmed = True
            except Exception:
                pass
        try:
            await self.open_mcp_sessions()
        except Exception as exc:  # noqa: BLE001 - omit-and-report
            self.missing.append(f"mcp sessions: {exc}")
        self._warmed = True
        return self

    async def _voice_fast(self, audio, sr: int = 16000, session_id: str | None = None):
        """Fast voice turn via the cascade. Yields events.

        ..
        """
        async for ev in self.cascade.run_fast(audio, sr, session_id=session_id):
            yield ev

    async def _voice_turns(self, audio, sr: int = 16000, session_id: str | None = None):
        """Full voice turn via the cascade. Yields events.

        ..
        """
        async for ev in self.cascade.run_turns(audio, sr, session_id=session_id):
            yield ev
