# 01 — Agent

The agent turns speech or text into work, and work into a spoken answer.
`src/agent/voice_agent/agent.py:VoiceAgent` is the shell: it inherits `LongRunningAgent`
(`src/agent/long_running/agent.py`) and owns the voice cascade
(`src/agent/voice_agent/cascade.py:CascadePipeline`). `src/server` exposes both over `:8004`.

Nothing here is aspirational. Every claim names the file that implements it.

## 1.1 Component map

```
voice-pipeline/
├── src/agent/
│   ├── voice_agent/
│   │   ├── agent.py             # VoiceAgent: text + voice entries
│   │   └── cascade.py           # CascadePipeline: VAD -> STT -> harness -> TTS
│   ├── long_running/
│   │   ├── agent.py             # LongRunningAgent facade: legs, stores, entries
│   │   ├── builder.py           # create_deep_agent + middleware order
│   │   ├── turn.py              # TurnRunner: box -> graph -> events
│   │   ├── box.py               # cwd jail + optional container
│   │   ├── worker.py            # warm-once LLM state
│   │   ├── scheduler.py         # admission + per-session serialization
│   │   ├── recall.py            # L2/L3 AgentMemory service
│   │   ├── messages.py          # LC message/reply helpers
│   │   ├── speech.py            # spoken caps + VRAM readout
│   │   ├── specialists.py       # planner -> 3 simulated specialists
│   │   ├── simtask.py           # depth-1 subtask cap (MAX_SUBTASKS = 6)
│   │   ├── episodic.py          # L2 episodic store
│   │   ├── middleware/          # session, memory, context, policy, trim, ...
│   │   ├── prompts/             # core, handoff, memory, sim, todo
│   │   └── AGENTS.md            # user-editable standing instructions
│   ├── prompts/                 # shared prompts: base, voice, terminal, chat, ...
│   ├── sandbox/                 # docker turn box
│   ├── events.py                # AgentEvent
│   ├── memory.py                # JsonSessionMemory
│   ├── store.py                 # SQLite sessions + tool spill
│   ├── facts.py                 # L3 facts
│   ├── budget.py                # context/step budgets
│   ├── title.py                 # two-word session titles
│   ├── tool_router.py           # op schemas + optional semantic shortlist
│   ├── shell.py                 # persistent shell + background jobs
│   └── mcp/                     # MCP config, client, servers
├── src/models/                  # VAD/STT/LLM/TTS + cloud providers + runtime
├── src/server/                  # FastAPI backend, :8004
└── src/tauri/                   # Tauri + SolidJS desktop client
```

## 1.2 `VoiceAgent`

`VoiceAgentConfig` (`src/agent/voice_agent/agent.py`) is split, not flat: `cascade`
(`CascadeConfig`), `harness` (`LongRunningConfig`) and `fast_voice`.
`VoiceAgent.__init__` builds the harness first, then wraps it with
`CascadePipeline`; `self.tts` is the shared TTS alias.

Harness defaults worth knowing (`LongRunningConfig`, `src/agent/long_running/agent.py`):

| Setting | Default | Effect |
|---|---|---|
| `work_dir` | `.` | shell/box root |
| `max_agent_steps` | 6 | recursion limit is `10 + 5 × steps` |
| `exec_timeout_s` / `tool_timeout_s` | 30 | shell + tool caps |
| `max_inflight` | 4 | FIFO admission across sessions |
| `max_session_turns` | 8 | L1 window size |
| `session_ttl_s` / `max_sessions` | 1800 / 1000 | session store limits |
| `worker_eager` | False | warm the LLM at boot |
| `use_tool_router` / `tool_top_k` | False / 5 | optional semantic tool shortlist |
| `memory_store` / `memory_recall` | False / False | L2 episodic + L3 facts |

Construction (`LongRunningAgent.__init__`) builds either a local `LlmModel` +
`LocalChatModel` or a stock LangChain model for a cloud backend, the
`LocalShellBackend`, MCP tools, `JsonSessionMemory`, `AgentMemory`,
`AdmissionQueue`, `TurnScheduler`, `TurnBox`, `WorkerLeg` and then a standby
graph. The per-turn graph is rebuilt by `TurnRunner` with `SessionMiddleware`.

### Warm order

`VoiceAgent.warm` (`src/agent/voice_agent/agent.py`) warms `vad`, `stt`, `tts`, `llm` in
that order. `tts` is skipped when `speak_text_turns` is false; `llm` is
skipped unless `worker_eager`. A failed leg is appended to `missing` and
warm continues, so `/health` can report `agent_loaded: false` with reasons
instead of refusing to boot. MCP sessions open last.

## 1.3 Turn paths

`VoiceAgent.__call__` (`src/agent/voice_agent/agent.py`) dispatches:

| Input | Path |
|---|---|
| `str` | `_text_reply` -> `TurnRunner.collect_text` -> reply string |
| audio, `fast_voice` | `CascadePipeline.run_fast` |
| audio | `CascadePipeline.run_turns` |

Text turns that need events use `LongRunningAgent.run_text`
(`src/agent/long_running/agent.py`) -> `TurnRunner.run_text`, which yields
`term/*` events and optional TTS audio. The plain `LongRunningAgent.__call__`
returns only the final reply.

Every turn serializes per session in `TurnScheduler.run`
(`src/agent/long_running/scheduler.py`): a second turn queues and is
injected as a follow-up instead of interleaving. `TurnRunner._run_turn`
ensures the worker first, then `TurnRunner._agent_invoke` opens the
`TurnBox`, rebuilds the graph, and streams the run. Session persistence
happens once in `SessionMiddleware.after_agent`, not in the runner.

### Lazy worker

`WorkerLeg.ensure` (`src/agent/long_running/worker.py`) warms the LLM on
first use. A refusal latches in `worker.unavailable`; later turns fail fast
with the same reason. `WorkerLeg.missing_reason` turns that latch or a
`missing[]` entry into the transcript notice instead of a silent empty
reply.

## 1.4 Harness build

`build_longrunning_agent` (`src/agent/long_running/agent.py`) delegates to
`builder.build_agent` (`src/agent/long_running/builder.py`), which returns a
compiled `CompiledStateGraph`. It rebuilds every turn, so state does not leak.
Layers, outermost first:

1. `TodoListMiddleware` — `write_todos` planning.
2. `SessionMiddleware` — inject stored history, persist the turn once.
3. `MemoryMiddleware` — L2 episodes + L3 facts in the system prompt.
4. `TurnContextMiddleware` — working directory + shell in the system prompt.
5. `TrimObservationsMiddleware` — observations capped at 1500 chars.
6. `PolicyMiddleware` — deny/confirm gates.
7. `ErrorHintMiddleware` — tool-failure hints.
8. `VerifyMiddleware` — result checks.
9. `InjectToolMiddleware` — optional top-K tool shortlist.
10. `SummarizationMiddleware` — always-on in-turn compaction.

The planner binds only the `Main` tool group
(`build_execution_tools(..., tags={"Main"})`) and either acts directly or
hands steps to subagents. The system prompt is the fixed core
(`src/agent/long_running/prompts/core.py`) plus the active-model identity line
plus the user-editable `src/agent/long_running/AGENTS.md`.

Before the graph runs, `TurnBox.open` (`src/agent/long_running/box.py`)
resolves the cwd and builds the per-turn backend (sandbox or local).
`TurnRunner._invoke_boxed` sends only the current user message; middleware
adds history, memory and `Working directory`/`Shell` to the model request.
It streams `messages` + `updates`. If the model did work but produced no
text, the reply is forced to `"Done."`; a degenerate reply is replaced
with `DEGENERATE_REPLY`.

## 1.5 Events

`AgentEvent` (`src/agent/events.py`) is the only currency the agent
speaks: `node`, `kind`, `data`, `t_s`. `as_dict` makes ndarrays JSON-safe.

| kind | Meaning |
|---|---|
| `queued` | another turn holds this session |
| `segments` / `text` | VAD spans, STT transcript |
| `token` / `thinking` | decoded piece, reasoning delta/block |
| `action` / `observation` | tool call, tool result |
| `deny` | policy block or user cancel |
| `chat` / `audio` | final reply, TTS chunk |
| `done` / `summary` | stream end, turn totals |
| `error` | leg or turn failure |

## 1.6 Planner -> specialists

`build_specialists` (`src/agent/long_running/specialists.py`) declares three
deepagents subagents over the **same** warmed model:

| Role | Tag | Tools |
|---|---|---|
| `shell-use` | `ShellUse` | terminal, files, Python, jobs |
| `browser-use` | `BrowserUse` | web search/read/operate |
| `computer-use` | `ComputerUse` | desktop GUI |

Specialization is prompt + tag-filtered tools + middleware, never extra
weights. Non-shell specialists also get a read-only filesystem allowlist
(`ls`, `read_file`, `glob`, `grep`). Each specialist carries its own
`PolicyMiddleware` + `TrimObservationsMiddleware` because deepagents does
not inherit parent middleware.

`create_deep_agent(subagents=...)` supplies the `task` transport: the
description is the specialist's only context, and its final text returns
as a `ToolMessage`. A specialist stack has no `task` tool, so recursion is
impossible by construction; `MAX_SUBTASKS = 6` caps breadth
(`src/agent/long_running/simtask.py`).

## 1.7 Memory, titles, budgets

`JsonSessionMemory` (`src/agent/memory.py`) owns L1: per-session history,
titles, locks and JSON persistence. When `memory_store` or `memory_recall`
is on, `EpisodicStore` (`src/agent/long_running/episodic.py`) stores evicted
turns as L2 and `load_facts` (`src/agent/facts.py`) supplies L3 facts.
`MemoryMiddleware` (`src/agent/long_running/middleware/memory.py`) appends
L2 + L3 to the system prompt; `AgentMemory.consolidate` promotes evicted
turn content after `SessionMiddleware` writes back.

Titles are two-word labels from the small leg (`src/agent/title.py`),
fired after the first turn by `_maybe_title` (`src/server/state.py:284`).
A dead leg or unparseable answer falls back to the client's offline name.

Budgets (`src/agent/budget.py`):

| Backend | Context | Step floor |
|---|---|---|
| `bonsai` | 16384 | 512 |
| `qwen` | 8192 | 256 |

`HEADROOM` is 0.8 and `CHARS_PER_TOKEN` 3.5; token counting prefers the
server's `/tokenize` when available. `TrimObservationsMiddleware`
(`src/agent/long_running/middleware/trim.py`) truncates oversized tool results
without mutating the original.

## 1.8 See also

- [02-tools.md](02-tools.md) — schemas, parser, router, policy
- [03-memory.md](03-memory.md) — L1/L2/L3 and injection
- [04-sessions.md](04-sessions.md) — ids, windows, persistence
- [05-models.md](05-models.md) — legs and `LlmModel`
- [06-processes.md](06-processes.md) — the `:8004` backend
- [07-inference.md](07-inference.md) — retired custom runtime
- [08-tui.md](08-tui.md) — the Tauri desktop client
