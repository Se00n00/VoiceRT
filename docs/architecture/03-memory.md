# 03 — Memory

Three tiers. One injection point. A write-back that only fires when something
actually fell out of the window.

## 3.1 The tiers

| Tier | Where | Store | Survives restart |
|---|---|---|---|
| L1 | in-window conversation | `LangChainSessionMemory` (RAM) / `JsonSessionMemory` (disk) | only with the JSON store |
| L2 | episodic episodes | `EpisodicStore`, sqlite + embeddings | yes |
| L3 | flat facts | `facts.md`, markdown | yes |

L1 is the live turn window. L2 is summarized history that aged out. L3 is
durable statements the agent decided to keep.

## 3.2 L1 — the window

`src/agent/memory.py` ships two implementations with the same surface.

### `LangChainSessionMemory` (`memory.py:89-230`)

RAM only, thread-safe behind a lock.

| Knob | Default | Where |
|---|---|---|
| `max_turns` | 20 | `memory.py` |
| `max_age_s` | 1800 (30 min) | `memory.py` |
| `max_sessions` | 1000 | `memory.py` |
| `max_tokens` | 0 (off) | `memory.py` |

Three eviction paths:

- **TTL** — `_sweep_locked` (`memory.py:116-119`) drops sessions older than
  `max_age_s` on every access.
- **LRU** — `_get_locked` (`memory.py:121-130`) evicts the oldest `at` stamp
  when a new session would exceed `max_sessions`.
- **size** — `_prune_locked` (`memory.py:132-149`) trims to `2 × max_turns`
  messages, and when `max_tokens > 0` also fills newest-first within the token
  budget while always keeping the last 2 messages.

Titles live in a side `_titles` dict (`memory.py:111`) with
`set_title`/`get_title` (`memory.py:199-219`).

### `JsonSessionMemory` (`memory.py:243-399`)

Same semantics, one file per session under `sessions_dir`. Writes are atomic —
tmp file plus `os.replace` (`memory.py:260-279`) — so a crash mid-write cannot
corrupt a session. Loading re-checks the TTL and `unlink`s stale files
(`memory.py:297-320`), and a sweep runs over the directory
(`memory.py:281-295`).

Session ids are sanitized by `_safe_sid` (`memory.py:232-240`) before they are
ever used as a filename.

## 3.3 L2 — episodic

`EpisodicStore` (`src/agent/episodic.py`) is sqlite plus embeddings.
`EMBED_MODEL_ID` is `bge-small-en-v1.5` (`episodic.py`).

`store()` (`episodic.py:100-122`) writes a summary and its blob embedding.
`recall()` (`episodic.py:134-176`) scans the last 2000 episodes, filters by
session and age, and returns top-k.

Ranking is a weighted blend (`episodic.py:171`):

```
score = 0.6 · max(cosine, 0)  +  0.3 · keyword_overlap  +  0.1 · recency
```

The keyword and recency terms exist so a semantically distant but lexically
identical episode ("renamed README.md") can still surface.

`summarize_turns` (`episodic.py:32-44`) is an extractive fallback summarizer,
used when no model is available to condense.

## 3.4 L3 — facts

`src/agent/facts.py` is deliberately dull: a markdown file.

- `load_facts` (`facts.py:23-55`) parses lines shaped
  `- subject — PREDICATE — object <!-- src,date -->`.
- `upsert_fact` (`facts.py:57-87`) replaces by `(subject, predicate)`,
  newest wins, written atomically.
- `format_block` (`facts.py:90-103`) renders a size-capped block anchored on
  `ANCHOR_LINE`.

## 3.5 Injection — the one place

`_memory_context` (`src/main.py:958-980`) is the single injection point. It
returns `""` when `memory_recall` is off, otherwise it assembles:

```
<L2 bullets>   episodic_store().recall(text, k=3, session)     main.py:965-971
<L3 block>     format_block(load_facts(memory_dir / facts.md)) main.py:973-977
```

That string is concatenated into the human message in `_agent_invoke`:

```python
content = str(text) + mem_ctx + f"\nCWD: {cwd} SHELL: bash"
```

at `src/main.py:648`.

Only three of the 199 earlier retrieval candidates are injected per turn. The
budget is small because the front brain's context is 4096 tokens on some
backends — recall that costs recall.

## 3.6 Write-back

`_memory_consolidate` (`src/main.py:938-956`) diffs the history contents before
and after `remember_turn`, identifies messages that were dropped by eviction,
summarizes them with `summarize_turns`, and stores the result into the episodic
store (`main.py:949-955`).

It is called only when `memory_store` is enabled (`src/main.py:761-769`).

So the loop is: turn runs → window is pruned → whatever fell out is summarized
once and promoted to L2 → next turn recalls up to 3 of them. Nothing is
promoted twice.

## 3.7 Known gaps

- **The `CWD:/SHELL:` trailer leaks into path arguments.** t16 of the own-tools
  eval produced `.SHELL:/.CWD/README.md`. The candidate fix is to move that
  context into the system prompt, then re-run the evals to measure the delta.
  Not done — the numbers would have to be re-measured.
- **Stale-title eviction.** `set_title`/`get_title` touch `_titles`
  independently of session LRU, so an evicted session can leave its title
  behind. Needs an audit.
- **Background title timing.** `/deep` reaches `title.py` through a different
  path than the primary turn and was not audited.

## 3.8 See also

- [01-agent.md](01-agent.md) — budgets that bound this
- [04-sessions.md](04-sessions.md) — window ownership and persistence
