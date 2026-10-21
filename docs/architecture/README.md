# Architecture

The technical architecture of `voice-pipeline`, split by part. Nothing here is
aspirational — each claim names the file that implements it, and open problems
are labeled as such rather than smoothed over.

## Reading order

| # | Part | What it covers |
|---|---|---|
| 00 | [ARCHITECTURE](../../ARCHITECTURE.md) | system design, contracts, dependency rules (single file) |
| 01 | [Agent](01-agent.md) | `VoiceAgent`, warm order, the turn loop, events, planner/specialists, budgets |
| 02 | [Tools](02-tools.md) | the execution ops, the five parser formats, semantic router, deny/confirm policy, MCP |
| 03 | [Memory](03-memory.md) | L1 window, L2 episodic, L3 facts, the one injection point, write-back |
| 04 | [Sessions](04-sessions.md) | identity, window ownership, persistence, serialization, resume |
| 05 | [Models](05-models.md) | the four legs, `LlmConfig`, backends, streaming, VRAM |
| 06 | [Processes](06-processes.md) | one `:8004` backend, endpoints, and the startup-warm trap |
| 07 | [Inference](07-inference.md) | retired (custom paged runtime deleted) |
| 08 | [TUI](08-tui.md) | the Tauri + SolidJS client, commands, voice mode, persistence |

## The short version

```
   mic ──▶ VAD ──▶ STT ──▶ VoiceAgent ──▶ TTS ──▶ speakers
                                 │
                    LongRunningAgent (deep agent)
                    planner + simulated specialists
                                 │
                       AgentEvent stream
                    (token/thinking/action/
                     observation/deny/summary)
```

One backend on `:8004`, one GPU process at a time on a 4 GB card:

```
voicert (Tauri)   ──:8004──▶ src.server   voice + terminal + TUI turns
console_cli       ─────────▶             text REPL (no GPU)
```

## The three things worth knowing

**The planner decides, the specialists act.** Multi-step work is planned with
`write_todos` and handed to tag-narrowed specialists; simple actions run
directly on the planner ([01](01-agent.md)).

**Enter never runs a command you did not type.** The TUI palette only hijacks
Enter after you move the highlight, because a palette that silently executes
`/model` when you typed `/mode` is a trap ([08](08-tui.md#enter-does-not-hijack)).

**A hung warm looks exactly like a dead backend.** Uvicorn awaits lifespan
startup before it binds, so a model download that hangs inside `_warm()` leaves
no listener at all ([06](06-processes.md#64-the-startup-warm-trap)).

## On-disk layout (XDG)

Seeded on first backend start (`src/app_config.py:AppConfig`,
paths in `src/paths.py:VoicertPaths`, store in `src/agent/store.py:VoicertStore`).
No legacy fallbacks: `configs/` and `.voicert/` are gone.

```
~/.config/voicert/          # global
├── skills/                 # all skills
├── mcp.yaml                # MCP servers
├── models.yaml             # providers + default model per backend
└── voicert.json            # permissions + voicert configs (edit: ask, …)

~/.local/share/voicert/     # local session
├── voicert.db              # mainstore — sessions + tool_outputs (sqlite)
├── voicert.log             # timestamped log
├── sessions/               # JSON session files (mirrored into voicert.db)
└── tool_outputs/           # spilled tool outputs over spill_chars (8000)
```

## Current state

| Suite | Result |
|---|---|
| TUI headless checks | 158 pass |
| TUI typecheck | clean |
| TUI build | 8 files → `dist/` |
| Python unittest | 634 run, 630 pass, 0 errors, 4 skipped |
| GPU | available, no model resident |

The 4 skips are intentional `@unittest.skipUnless` gates (CUDA/live-voice
tests). See [07](07-inference.md) for the retired runtime note.

## Companion docs

- [`benchmarks.md`](../benchmarks.md) — the eval harnesses and measured results
- [`capacity.md`](../capacity.md) — how generation length prices sessions
- [`../README.md`](../../README.md) — the voice pipeline
