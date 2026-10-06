# ARCHITECTURE — entities and how they work

Start here. The part files below it (`architecture/01–08`) hold the
file-and-line detail; this page is the map: every entity, what it owns,
and how a voice turn flows through them.

> The Mermaid blocks below render on GitHub but not in Zed's preview,
> so each one is followed by a plain-ASCII twin that renders anywhere.

```mermaid
flowchart TD
    User["User: speaks, types, interrupts"]
    VoiceAgent["VoiceAgent: warm, run_text, call, locks, confirm gate"]
    VadModel["VadModel: segments, active"]
    SttModel["SttModel: transcribe, partial"]
    FrontBrain["FrontBrain Qwen3-0.6B: escalate, chat, no tools"]
    WorkerAgent["WorkerAgent Bonsai 27B: delegate, tools, memory"]
    LocalChatModel["LocalChatModel: prompt, stream"]
    TtsModel["TtsModel: speak, voices"]
    ToolRouter["ToolRouter: shortlist, policy"]
    Memory["Memory: recall, consolidate"]
    Sessions["Sessions: history, titles"]
    Bridge["Bridge 8004: runText, confirm"]
    Server["Server 8003: talk, barge"]
    TUI["TUI: render, playback"]
    User -->|text audio cancel| VoiceAgent
    VoiceAgent -->|speech spans| VadModel
    VoiceAgent -->|transcript| SttModel
    VoiceAgent -->|route chat delegate| FrontBrain
    VoiceAgent -->|delegated task| WorkerAgent
    WorkerAgent -->|tokens| LocalChatModel
    WorkerAgent -->|allowed ops| ToolRouter
    WorkerAgent -->|recall consolidate| Memory
    VoiceAgent -->|remember turn| Sessions
    VoiceAgent -->|speak replies| TtsModel
    TtsModel -->|audio| User
    Bridge -->|run_text confirm| VoiceAgent
    Server -->|audio turns barge| VoiceAgent
    TUI -->|turns over WS| Bridge
```

ASCII twin (same entities, same edges):

```
User ──text/audio/cancel──▶ VoiceAgent ──speech spans──▶ VadModel
   ▲                            │  transcript            (Silero)
   │                            ▼
   │                     ┌──────────────┐
   │                     │ SttModel     │──partials while speaking
   │                     │ (Whisper)    │  (prefill the LLM cache)
   │                     └──────┬───────┘
   │                            │ route?
   │              ┌─────────────┴──────────────┐
   │              ▼                            ▼
   │     FrontBrain (Qwen3-0.6B)      WorkerAgent (Bonsai 27B)
   │     YES/NO + short chat          tools + router + memory
   │     zero tools                   confirm-gated
   │              │                            │
   │              │   ┌────────────────────────┘
   │              ▼   ▼
   │     LocalChatModel (prompt + parse + tokens)
   │                            │
VoiceAgent ◀── remember_turn ───┤
   │                            ▼
   │                     TtsModel (Kokoro, per sentence)
   │                            │ audio (interruptible: barge restarts VAD)
   └────────────────────────────┘

Transports:  TUI ──WS──▶ Bridge (:8004, run_text + confirm)
             mic ──WS──▶ Server (:8003, loop + partials + barge)
Shared:      ToolRouter (allow/deny policy) · Memory (L1/L2/L3) · Sessions
```

One voice turn, in order:

```mermaid
sequenceDiagram
    participant U as User
    participant V as VAD
    participant S as STT
    participant F as Front06B
    participant W as Worker27B
    participant T as TTS
    U->>V: mic frames stream
    V->>V: speech check keep buffering
    Note over V: trailing pause ends turn, VAD 0.06s
    V->>S: trimmed speech, first token 52ms steady
    S->>W: word deltas prefill during speech
    Note over S: 6.1ms per token after, partials every 0.7s
    S->>F: transcript plus rolling partials
    F->>F: YES NO escalate 0.96s
    alt chat
        F->>T: short reply 0.27s
    else delegate
        F->>W: decode from cached prefix
        Note over W: pause to first token is one decode step
        W->>T: token groups synth per chunk
        Note over T: TTS 0.7s per chunk, first audio target sub second
    end
    T->>U: audio barge restarts VAD
```

Measured 2026-10-04 on this box (RTX 3050 4GB): VAD 0.06s, STT
TTFT 52ms steady with 6.1ms per token after, front decide 0.96s,
front chat 0.27s, worker LLM 76s via the gemma-4B CPU sidecar,
TTS 0.7s per chunk, end-to-end 78.3s. Streaming prefill (word deltas
prime the decode leg, abort on revision) and barge-in are live in
server.py — except the sidecar HTTP prime, which stays OFF: a prime
costs 2.4s of prompt processing on the single-slot CPU sidecar, longer
than the pause window, and would hold the only slot while the real
turn waits. Revisit when the worker has slots to spare or runs fused.

ASCII twin (same order, same branches):

```
User ──mic frames (stream)──▶ VAD ──speech? keep buffering──▶ VAD
                                                      │
                               pause ends turn · VAD 0.06s
                                                      ▼
                               STT first token 52ms steady
                               6.1ms per token · partials every 0.7s
                               word deltas prefill the worker
                                                      │
                                               ┌───────┴────────┐
                                               ▼                ▼
                                      Front 0.6B          decode from cache
                                      YES/NO 0.96s        at silence: one step
                                               │          to first token
                               chat: short     │
                               reply 0.27s ──┐ │     token groups synth
                                       ▼       ▼     per chunk 0.7s
                                      TTS ◀── first audio target sub second
                                       │
                                       ▼
                               User (audio; interrupt = barge → back to VAD)

Measured 2026-10-04, RTX 3050 4GB. Prefill, cached decode and sub-second
first audio are targets being built; pause to first audio is the number
this design minimizes.
```

## The entities, briefly

| Entity | What it does | Detail |
|---|---|---|
| `User` | speaks, types, hits Ctrl+C | [TUI](architecture/08-tui.md) |
| `VoiceAgent` (`src/main.py`) | owns legs, locks, queue, confirm gate, turn loop | [01](architecture/01-agent.md) |
| `VadModel` (Silero ONNX) | speech spans, live per-chunk test, endpointing | [05](architecture/05-models.md) |
| `SttModel` (Whisper) | full + rolling-partial transcription | [05](architecture/05-models.md) |
| `FrontBrain` (Qwen3-0.6B) | one YES/NO + short chat replies, zero tools | [01](architecture/01-agent.md) |
| `WorkerAgent` (Bonsai 27B) | deep-agent loop: tools, router, memory, subagents | [01](architecture/01-agent.md), [02](architecture/02-tools.md) |
| `LocalChatModel` | prompt assembly, five parse formats, token stream | [05](architecture/05-models.md) |
| `TtsModel` (Kokoro) | per-sentence synth, voice select, overlap play | [05](architecture/05-models.md) |
| `ToolRouter` | semantic op shortlist + deny/confirm/allowlist policy | [02](architecture/02-tools.md) |
| `Memory` | L1 window injection, L2 episodes, L3 facts | [03](architecture/03-memory.md) |
| `Sessions` | per-session history, titles, locks, resume | [04](architecture/04-sessions.md) |
| `Bridge` (:8004) | terminal + autonomous sockets, confirm mailbox | [06](architecture/06-processes.md) |
| `Server` (:8003) | mic loop, partials, prefill, barge-in, latency | [06](architecture/06-processes.md) |
| `TUI` (`voicert`) | panels, palette, mic viz, playback, interrupt | [08](architecture/08-tui.md) |

Rules that cross entities: one GPU process at a time · `PYTHONPATH=.`
always · confirm before mutating tools · sys logs never in the transcript.
