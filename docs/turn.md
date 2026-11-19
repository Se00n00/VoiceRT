# One turn, start to finish

What happens between a microphone buffer and a speaker buffer. Every claim
names the file that implements it; numbers live in [benchmarks.md](benchmarks.md).

## 1.1 Pick a lane

| Lane | Trigger | Model work | First audio |
|---|---|---|---|
| Full agent | `VoiceAgent(audio)` default (`fast_voice=False`) → `CascadePipeline.run_turns` (`src/agent/voice_agent/cascade.py:381`) | planner + tools + specialists | after the first clause of the final reply |
| Fast lane | `fast_voice=True` → `run_fast` (`cascade.py:155`) | one `llm.generate`, no tools | after the whole reply |
| Duplex call | `POST /talk/offer` → `TalkCall.run_utterance` (`src/server/services/talk.py:176`) | one `llm.generate`, 256 tokens | streamed sentence by sentence |

The duplex call owns its own loop (barge-in, 24 kHz out, no admission
ticket); the other two share the cascade and its event shapes.

## 1.2 The full-agent turn

```
 audio float32 @ 16 kHz
   │
   │  budget guard        check_budget(per_turn_mb, vram_budget_mb)   cascade.py:392
   │  admission ticket    AdmissionQueue.admit()                      scheduler.py:18
   ▼
 VAD   silero segments    vad.segments()                             vad.py:116
   │   no speech → empty stt text + turn/summary, return             cascade.py:425
   ▼
 STT   whisper-base       stt.transcribe()                           stt.py:69
   │   no text  → same early exit                                    cascade.py:463
   ▼
 AGENT TurnRunner.run     graph + tools + memory                     turn.py:293
   │   llm/token · llm/thinking · action · observation · …           cascade.py:492
   │   out_turns collects (text, reply, thinking) per response
   ▼
 TTS   splitter + clause fan-out                                    cascade.py:61
   │   synthesize the first clause, not the first sentence           cascade.py:531
   ▼
 turn/summary  node_s · ttfa_s · total_s · vram_mb · session_id      cascade.py:575
```

Two timing facts worth remembering:

- **Tokens stream before audio.** The harness streams `llm/token` events while
  the graph runs, but synthesis starts only after the reply is complete; the
  wait is hidden because the first *clause* is synthesized instead of the
  whole reply.
- **Silence is a successful turn.** No VAD segments or empty STT text yields
  `stt/text` with `silent: True` (or empty text) plus a summary. No LLM call,
  no TTS, no error.

## 1.3 The event contract

All nodes emit `AgentEvent` (`src/agent/events.py:19`); `as_dict()` makes the
payload JSON-safe and summarizes ndarrays instead of serializing them.

| Node | Kind | Payload that matters |
|---|---|---|
| `vad` | `segments` | `segments`, `audio_dur_s`, `speech_s` |
| `stt` | `text` | `text`, `rtf`, `ttfs`, `dur_s`; `silent` when the clip is empty |
| `llm` | `token` / `thinking` / `done` | `token`, `piece`, `first`; think text; final `text` |
| harness | `action` / `observation` / `confirm` / `deny` / `queued` / `route` / `error` | passed through untouched by the cascade |
| `tts` | `audio` | `wav`, `sr`, `sentence`, `synth_s` — one per clause |
| `turn` | `summary` | `text`, `reply`, `node_s`, `ttfa_s`, `total_s`, `vram_mb`, `session_id` |

`node_s` is the per-leg stopwatch, accumulated so a retry inside a leg does not
hide its full cost.

## 1.4 Four details that matter

1. **Two guards before a sample is processed.** The VRAM budget check runs
   first (`cascade.py:392`), then a FIFO admission ticket is taken
   (`AdmissionQueue`, `src/agent/long_running/scheduler.py:18`). A turn that
   waits past `queue_timeout_s` fails with `server saturated (…)`, which is
   counted in `/metrics`.
2. **No ticket, no history.** The session is remembered only when the caller
   supplied a `session_id` (`remember = bool(session_id)`); anonymous audio
   turns stay stateless.
3. **The spoken form is capped, the recorded one is not.** `cap_words` trims
   each reply before synthesis and `llm/done` carries that capped text; the
   final `turn/summary` reports the raw last reply from `out_turns`.
4. **The graph is rebuilt every turn.** `TurnRunner` builds a fresh
   `CompiledStateGraph`, so state cannot leak between turns; only the session
   store and memory survive.

## 1.5 Inspect a live turn

`GET /metrics` exposes queue and turn counters; the `turn/summary` event is
the per-turn receipt with leg timings. For the duplex path, the data channel
carries `transcript`, `barge-in`, and state transitions instead.

## See also

- [architecture/01-agent.md](architecture/01-agent.md) — harness, warm order, specialists
- [architecture/04-sessions.md](architecture/04-sessions.md) — session identity and persistence
- [APIReferences/Voice.md](APIReferences/Voice.md) — `/term/stt`, `/term/say`
- [APIReferences/Talk.md](APIReferences/Talk.md) — `/talk/offer`
