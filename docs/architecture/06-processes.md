# 06 — Processes

One backend on `:8004`, plus a TUI client and a GPU-free text console.

## 6.1 Which process is which

| Command | Port | Purpose |
|---|---|---|
| `PYTHONPATH=. .venv/bin/python -m src.server` | 8004 | FastAPI backend (voice + text + TUI) |
| `PYTHONPATH=. .venv/bin/python -m src.server.console_cli` | — | text REPL (no GPU) |
| `cd src/tauri && npm run dev` | — | Tauri + SolidJS client |

**Run one at a time.** Two agents = two weight copies = CUDA OOM on a 4 GB
card.

## 6.2 `src/server` — the backend

Endpoints (`src/server/routes/`):

| Route | File | Purpose |
|---|---|---|
| `GET /health` | `system.py` | liveness, warm state, missing legs |
| `GET /model` | `system.py` | current model profile |
| `POST /model/switch` | `system.py` | hot-swap the LLM leg |
| `GET /metrics?sid=`, `/brain` | `system.py` | counters, health, context and active legs |
| `GET /term/title` | `sessions.py` | session title |
| `GET /term/history` | `sessions.py` | past session transcript |
| `POST /term/stt` | `voice.py` | audio → text |
| `WS /term/stt-stream` | `voice.py` | live partial transcripts |
| `POST /term/say` | `voice.py` | text → audio |
| `WS /term` | `turns.py` | the terminal turn stream |
| `WS /deep` | `turns.py` | the deep-agent turn stream |
| `WS /chat` | `chat.py` | chat turn stream |
| `GET/POST/DELETE /keys` | `keys.py` | provider key storage |
| `POST /talk/offer` | `talk.py` | WebRTC voice offer |

Supporting pieces (`src/server/state.py`): `_build_llm`, `_free_llm_gpu`,
`_maybe_title`, `_compact_sessions`, `switch_model`, `get_agent`,
`_wav_b64`; `create_app` lives in `src/server/__init__.py`.

`/term` runs one turn at a time per session and keeps a pending queue with a
`Queue` plus per-turn `Event`s (`turns.py`, timing out at 120 s).
Turn work is dispatched with `asyncio.create_task`.

`/model/switch` to an explicitly warmed leg resets the lazy-worker latch
(`state.py: switch_model`); if the new sidecar only attached to the old server,
it re-warms after the old close so the swap never commits a dead port.

`/health` must stay liveness-safe — it answers before the agent exists, which
is what lets the TUI show `backend down` versus `warming` correctly.

## 6.3 The separate voice API is gone

`bridge.py` and `server.py` were merged into `src/server` on `:8004`; there is
no `:8003` process. The old contact/Telegram/WhatsApp routes were retired.

## 6.4 The startup-warm trap

This one bit us and is worth writing down.

```python
@app.on_event("startup")
async def _warm():        # src/server/__init__.py
    await agent.warm()
```

Uvicorn **awaits the lifespan startup hook before it creates the listening
socket**. So a warm that hangs means no listener at all — not a slow start, a
silent port.

Observed on 2026-09-23: four backend processes, `nvidia-smi` showing no compute
process, six HTTPS sockets in `CLOSE_WAIT` against what resolves to a
HuggingFace CDN, and `:8004` refusing connections. The warm was stuck
downloading model weights from inside the lifespan hook.

The fix is to bind first and warm second — a background task plus an explicit
`warming` / `warmed` / `warm_error` state on `/health`, and a bounded warm so a
hung download cannot wedge startup. **Not implemented yet.** Until it is, a
hung warm looks exactly like a dead backend to the TUI.

## 6.5 Process hygiene

Model subprocesses (`arecord`, `aplay`, docker clients) die with their parents.
When in doubt:

```bash
ps aux | grep -E 'console_cli|test_toolcall|src.server'
```

## 6.6 See also

- [01-agent.md](01-agent.md) — what `create_app` wraps
- [08-tui.md](08-tui.md) — the client
