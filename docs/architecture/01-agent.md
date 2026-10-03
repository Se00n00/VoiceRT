# 01 — Agent

The agent is the thing that turns a sentence of audio (or text) into work, and
work into a spoken answer. It lives in `src/main.py` and is fronted by two thin
processes (`bridge.py`, `server.py`).

Nothing here is aspirational. Every claim names the file that implements it.

## 1.1 Component map

```
voice-pipeline/
├── src/main.py                 # VoiceAgent — the turn loop (1429 lines)
├── src/agent/
│   ├── events.py               # AgentEvent: the one currency (38 lines)
│   ├── prompts.py              # TERMINAL_PREAMBLE: the tool contract (12)
│   ├── chat_model.py           # LocalChatModel: LC BaseChatModel over LlmModel (550)
│   ├── delegate.py             # two-brain routing (642)
│   ├── title.py                # two-word session titles (144)
│   ├── budget.py               # context/step budgets per backend (273)
│   ├── trim.py                 # TrimObservationsMiddleware (53)
│   ├── episodic.py             # L2 episodic store, sqlite + embeddings (186)
│   ├── facts.py                # L3 flat facts, markdown (103)
│   ├── simtask.py              # depth-1 Sim-Agent subagents (259)
│   ├── tool_router.py          # bge-small semantic tool selection (206)
│   ├── memory.py               # LangChainSessionMemory / JsonSessionMemory (399)
│   ├── contacts.py             # telegram / baileys / callmebot legs (666)
│   ├── pairing.py              # Telegram creator-token pairing (395)
│   ├── tg_poll.py              # long-poll loop (79)
│   └── mcp/                    # MCP config + client + server
├── src/tools/terminal.py       # op schemas, policy regexes, parsers
├── bridge.py                   # TUI backend, :8004
└── server.py                   # voice API, :8003
```

## 1.2 `VoiceAgent`

`VoiceAgent` holds the four pipeline legs plus the agent harness. Config is
`VoiceAgentConfig` (`src/main.py:69-145`); defaults worth knowing:

| Setting | Default | Where |
|---|---|---|
| `max_agent_steps` | 6 | `main.py:90` |
| `recursion_limit` | 10 (+5×steps) | `main.py:91`, `main.py:661` |
| `tool_timeout_s` | 30 | `main.py:92` |
| `tool_out_cap` | 6000 chars | `main.py:93` |
| `exec_timeout_s` | 30 | `main.py:95` |
| `worker_eager` | False (lazy worker) | `main.py:144` |

Legs held after construction: `vad`, `stt`, `llm`, `tts`, plus `front_llm` when
delegation is on (`main.py:244-250`), `chat_model` (`main.py:253`), a
deepagents `LocalShellBackend` (`main.py:257-261`), `sessions`
(`main.py:269-278`), a lazy `SemanticRouter` (`main.py:279+`), a lazy
episodic store (`main.py:1000+`) and a `FIFOScheduler` (`main.py:281-286`).
Lazy-worker flags live with them (`main.py:291-299`): `_worker_eager`,
`_worker_warmed`, `_worker_unavailable`.

### Warm order

`warm()` (`main.py:398-437`) iterates legs in a fixed order — `vad`, `stt`,
`tts`, `front_llm`, `llm` (`main.py:407-425`). `tts` is skipped when
`speak_text_turns` is false, and the worker `llm` leg is skipped when
delegation is on without `worker_eager` — it boots on the first delegate
route instead (`_ensure_worker`, `main.py:642-671`), so boot carries zero
worker footprint. A refused on-demand warm latches in
`_worker_unavailable` and later YES turns fail fast with the same reason;
an explicit `/model` switch to a warmed leg resets the latch.

**One leg failing is never fatal.** Each exception is appended to `missing[]`
and warm continues, so `/health` can answer `agent_loaded: false` with a
reason list instead of the process refusing to boot.

## 1.3 The turn

`__call__` (`main.py:1105-1127`) dispatches on input type:

| Input | Path | Method |
|---|---|---|
| `str` | text reply | `_text_reply` (`main.py:1129+`) |
| audio, fast voice | single-shot | `_voice_fast` (`main.py:1201+`) |
| audio, full | multi-leg | `_voice_turns` (`main.py:1363+`) |

Every turn goes through `_run_locked` (`main.py:836-875`), which serializes per
`session_id` via a lock. Contenders are queued into `_pending` rather than
dropped (`main.py:852-855`), and the queue drains when the lock frees
(`main.py:868-875`).

`_run_turn` (`main.py:877-959`) is where the front brain is consulted. If the
front model says *chat*, the answer is emitted directly and no worker runs. If
it says *delegate*, `_ensure_worker` (`main.py:642-671`) warms the lazy worker
first — a refusal ends the turn with the reason instead of running anything —
then `_agent_invoke` (`main.py:682-835`) drives the full harness.

## 1.4 Events

`AgentEvent` (`src/agent/events.py:17-38`) is the only currency the agent
speaks. Fields: `node` (`vad|stt|llm|tts|turn`), `kind`, `data`, `t_s`
(`perf_counter`).

`as_dict()` (`events.py:24-38`) copies `data` and converts any numpy array to
`{"dtype","shape","n"}` so an event is always JSON-serializable.

Kinds emitted by the agent loop:

| kind | Emitted at | Meaning |
|---|---|---|
| `queued` | `main.py:853` | another turn holds this session |
| `route` | `main.py:904` | which brain answered |
| `token` | `main.py:738` | one decoded token |
| `thinking` | `main.py:747,771,815` | `<think>` delta or block |
| `action` | `main.py:785` | a tool call was made |
| `observation` | `main.py:806` | tool result |
| `deny` | `main.py:798,802` | policy block or user cancel |
| `chat` | `main.py:1089` | final text reply |
| `audio` | `main.py:1094` | TTS chunk |
| `error` | `main.py:705,809,821,885,1085` | turn failed |
| `summary` | `main.py:1100` | turn totals |

## 1.5 Worker invocation

`_agent_invoke` (`main.py:682-835`) is the interesting method:

1. **Rebuilds the agent every turn** via `_build_agent` (`main.py:700-708`).
    State does not leak between turns.
2. Builds the human message as
   `text + mem_ctx + "\nCWD: <cwd> SHELL: bash"` (`main.py:709-713`).
3. Streams with `agent.astream(..., ["messages","updates"])` and a
   `recursion_limit` of `10 + 5 × max_agent_steps` (`main.py:725-727`).
4. `messages` mode → token and thinking events. `updates` mode → inspects the
   payload node's messages to decide whether it is thinking, a tool call
   (`main.py:775-789`), or the final reply (`main.py:790-792`).
5. Finally, `split_thinking` (`main.py:811-819`) separates reasoning from
   answer. **If the worker did work but produced no text, the reply is forced
   to `"Done."`** (`main.py:818-819`).

That last rule is a deliberate scar tissue. Small models routinely narrate
("I renamed the file") and emit nothing; rather than hand an empty string to
TTS, the loop substitutes a real word.

## 1.6 Two-brain delegation

Config: `configs/delegate.yaml`, loaded by `src/agent/delegate.py:112-172`.
Selection order is explicit path → `$VOICE_DELEGATE_CONFIG` → the repo file →
built-in defaults.

| Brain | Model | Tools | Job |
|---|---|---|---|
| front | `Qwen/Qwen3-0.6B` | **none** | one yes/no routing question per turn |
| worker | `prism-ml/Ternary-Bonsai-2-27B` | all | the entire agentic harness |

The front brain gets no tools *by design*. It is asked one binary question, so
it cannot fabricate a tool call or invent a task string. `BOOLEAN_MAX_TOKENS`
is 6 (`delegate.py:346`) and `thinking: false`, because a trace only burns
tokens on a choice that has two answers.

`decide_escalate` (`delegate.py:383-402`) **fails toward escalation**: an
unparseable answer or an exception returns `True`. Silence means "go do the
work", which is the safe direction for a 0.6B model.

`backstop: true` adds a deterministic regex pass (`configs/delegate.yaml`,
`backstop_min_score: 2`) consulted *only* when the flag said no and the text
still reads like work. It costs a regex match and guards the worst bug this
design can produce: the assistant claiming to have done something it did not
do. Re-measured 2026-10-03 on the committed 28-prompt set
(`tests/src/test_delegate.py` CHAT+TASK): 13/13 tasks escalate, 11/15
chit-chat kept, and the model never said `NO` to anything the regex scored
as work — so the backstop still never fires.

Depth is capped at 1 — a subtask that emits its own plan runs flat, never
nested (`src/agent/simtask.py:13-15`, `MAX_SUBTASKS = 6`).

The worker boots lazily: `worker_eager: false` (default, `main.py:144`,
mirrored in `configs/delegate.yaml`) skips the `llm` leg at boot, and the
first delegate route warms it via `_ensure_worker` (`main.py:642-671`). A
refused warm latches in `_worker_unavailable`, so later YES turns fail fast
with the same reason; an explicit `/model` switch to a warmed leg resets
the latch (`bridge.py:345-349`).

When the worker leg is missing, `_worker_missing_reason`
(`main.py:986-997`) inspects `missing[]` for an `llm leg:` entry — and the
lazy latch first — and the turn carries that reason instead of a silent
empty answer (`main.py:936-959`).

## 1.7 Titles

`src/agent/title.py` generates a two-word session name with the front brain:
`TITLE_MAX_TOKENS = 12` (`title.py:59`), no tools, no history, user text
capped at 600 chars (`title.py:77-89`).

`parse_title` (`title.py:92-121`) strips `<think>`, drops a leading
narration preamble when a colon-narration pattern is detected (`_is_preamble`),
extracts at most two words, and caps length at 40. `generate_title`
(`title.py:124-145`) never raises — it returns `(title, reason)`.

## 1.8 Budgets

`src/agent/budget.py` prices every turn:

| Backend | Context tokens | Step floor |
|---|---|---|
| `bonsai` | 16384 | 512 |
| `gemma` | 4096 | — |
| `gemma270` | 32768 | 256 |
| `qwen17` | 32768 | 512 |

`HEADROOM` 0.8 and `CHARS_PER_TOKEN` 3.5 (`budget.py:57-59`). Token counting
prefers the server's real `/tokenize` when a URL is available
(`budget.py:158-173`) and falls back to the character estimate.
`window_slice` (`budget.py:193-273`) fills newest-first within budget and
always keeps at least one message.

`TrimObservationsMiddleware` (`src/agent/trim.py:14-53`) truncates any
`ToolMessage` over 1500 chars to head + `…[N chars omitted]`, and it does so
via `override`, never mutating the original.

## 1.9 See also

- [02-tools.md](02-tools.md) — the op schemas, parser, router, policy
- [03-memory.md](03-memory.md) — the three tiers and injection
- [04-sessions.md](04-sessions.md) — ids, windows, persistence
- [05-models.md](05-models.md) — the legs and `LlmModel`
- [06-processes.md](06-processes.md) — bridge, server, and their endpoints
- [07-inference.md](07-inference.md) — the custom runtime (parser fixed 2026-10-03)
- [08-tui.md](08-tui.md) — the OpenTUI client
