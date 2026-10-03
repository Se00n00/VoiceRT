# Architecture

The technical architecture of `voice-pipeline`, split by part. Nothing here is
aspirational — each claim names the file that implements it, and open problems
are labeled as such rather than smoothed over.

## Reading order

| # | Part | What it covers |
|---|---|---|
| 01 | [Agent](01-agent.md) | `VoiceAgent`, warm order, the turn loop, events, two-brain routing, budgets |
| 02 | [Tools](02-tools.md) | the twelve ops, the five parser formats, semantic router, deny/confirm policy, MCP |
| 03 | [Memory](03-memory.md) | L1 window, L2 episodic, L3 facts, the one injection point, write-back |
| 04 | [Sessions](04-sessions.md) | identity, window ownership, persistence, serialization, resume |
| 05 | [Models](05-models.md) | the four legs, `LlmConfig`, backends, streaming, VRAM |
| 06 | [Processes](06-processes.md) | bridge, server, their endpoints, and the startup-warm trap |
| 07 | [Inference](07-inference.md) | the custom runtime (parser fixed 2026-10-03, bench re-run) |
| 08 | [TUI](08-tui.md) | the OpenTUI client, command palette, shades, exit card |

## The short version

```
   mic ──▶ VAD ──▶ STT ──▶ VoiceAgent ──▶ TTS ──▶ speakers
                                 │
                    ┌────────────┴────────────┐
                    │  front brain 0.6B       │  no tools; one yes/no question
                    │  worker   Bonsai 27B    │  all tools, router, memory, subagents
                    └─────────────────────────┘
                                 │
                       AgentEvent stream
                    (token/thinking/action/
                     observation/deny/summary)
```

Three processes, never two at once on a 4 GB card:

```
voicert (OpenTUI)  ──:8004──▶  bridge.py     terminal turns
server.py          ──:8003──▶  (voice API)   WS /talk, contacts
llm_chat.py        ───────────▶               text REPL
```

## The three things worth knowing

**The front brain fails toward escalation.** A 0.6B model asked one yes/no
question is untrustworthy, so an unparseable answer means "go do the work"
([01](01-agent.md#16-two-brain-delegation)).

**Enter never runs a command you did not type.** The TUI palette only hijacks
Enter after you move the highlight, because a palette that silently executes
`/model` when you typed `/mode` is a trap ([08](08-tui.md#enter-does-not-hijack)).

**A hung warm looks exactly like a dead bridge.** Uvicorn awaits lifespan
startup before it binds, so a model download that hangs inside `_warm()` leaves
no listener at all ([06](06-processes.md#64-the-startup-warm-trap)).

## Current state

| Suite | Result |
|---|---|
| TUI headless checks | 158 pass |
| TUI typecheck | clean |
| TUI build | 8 files → `dist/` |
| Python unittest | 634 run, 630 pass, 0 errors, 4 skipped |
| GPU | available, no model resident |

The 4 skips are intentional `@unittest.skipUnless` gates (CUDA/live-voice
tests). The suite previously read 564 run with 14 errors, all from one
indentation slip at `src/inference/engine.py:431` — fixed 2026-10-03. See
[07](07-inference.md).

## Companion docs

- [`benchmarks.md`](../benchmarks.md) — the eval harnesses and measured results
- [`capacity.md`](../capacity.md) — how generation length prices sessions
- [`inference_engine.md`](../inference_engine.md) — the runtime design in depth
- [`../README.md`](../../README.md) — the voice pipeline
