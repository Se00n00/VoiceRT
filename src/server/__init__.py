"""Terminal bridge: FastAPI service for the Ink TUI (stock React/Ink).

Runs the unified VoiceAgent (voice + terminal turns over one deepagents
loop) and exposes it over HTTP + WebSocket on :8004. Run ONE backend
process (same 4GB GPU).

Endpoints (see ``src/server/routes/`` for the handlers):
- ``GET /health`` — {ok, agent_loaded, missing}
- ``GET /model`` — {current, available:[{name, model, backend, desc, ready}]}
- ``POST /model/switch`` {name} — hot-swap the LLM leg (VRAM-safe)
- ``POST /term/stt`` {pcm_b64, sr} — raw mic PCM -> {text}
- ``WS /term/stt-stream`` — live mic PCM16 mono frames in, partial/final
  transcripts out
- ``POST /term/say`` {text} — text -> {wav_b64, sr} (Kokoro)
- ``GET /term/history?sid=`` — past session messages
- ``GET /term/title?sid=`` — persisted session name (no model call)
- ``WS /term`` — full turn + confirm gate
- ``WS /deep`` — autonomous turns over the same agent (confirms auto-pass)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import state
from .routes import sessions_router, system_router, turns_router, voice_router

__all__ = ["create_app", "app"]


def create_app(agent=None):
    if agent is not None:
        state._agent = agent
    app = FastAPI(title="voice-term bridge")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(sessions_router)
    app.include_router(voice_router)
    app.include_router(turns_router)

    @app.on_event("startup")
    async def _warm():
        ag = state.get_agent()
        try:
            await ag.warm()
            print("bridge ready", flush=True)
            if getattr(ag, "missing", []):
                print(f"bridge missing: {ag.missing}", flush=True)
        except Exception as exc:
            print(f"bridge not warmed: {exc}", flush=True)

    return app


app = create_app()
