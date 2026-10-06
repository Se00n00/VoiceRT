# 02 — Tools

What the agent can actually do, how a model's text becomes a call, and the two
policy gates in front of execution.

## 2.1 The ops

`ALLOWED_OPS` (`src/tools/terminal.py:41`) — twelve, and `done` is not really a
tool, it is the "I am finished talking" sentinel:

```
exec  exec_bg  poll  read  write  edit  grep  list  python_exec  fetch  searxng  done
```

`TERMINAL_TOOLS` (`terminal.py:47-109`) declares eleven JSON-Schema specs for
models whose chat template supports tools natively (Qwen renders them into
`<tools>` XML, Bonsai takes OpenAI-style `tools[]`).

| Op | Params | Notes |
|---|---|---|
| `exec` | `command` | blocking bash, `exec_timeout_s` 30 |
| `exec_bg` | `command` | returns a job id |
| `poll` | `command` | job id from `exec_bg` |
| `read` | `path` | |
| `write` | `path`, `text` | create or overwrite |
| `edit` | `path`, `anchor`, `text` | anchored replace; **fails if the anchor is missing** |
| `grep` | `pattern`, `path?` | |
| `list` | `path?` | |
| `python_exec` | `code` | isolated container |
| `fetch` | `path` (a URL) | page as text |
| `searxng` | `pattern` | search |

The same three capabilities are also reachable over MCP
(`src/agent/mcp/tools.py:25-30`): `python_exec(code, timeout)`,
`fetch(url, max_chars, timeout)`, `web_search(query, n, timeout, provider)`.

## 2.2 Parsing a call out of text

Small models do not reliably emit one canonical format, so the parser accepts
five and tries them in order. `_parse_action` (`src/agent/chat_model.py:151-178`):

```
gemma_action → functiongemma_action → toolcall_dict → terminal_action → xml_action
```

each tried on the cleaned `answer`, then on the raw stream as a fallback.

The accepted shapes:

| Shape | Example |
|---|---|
| terminal JSON | `{"action": "exec", "command": "ls"}` |
| XML function | `<function name="read"><param name="path">x</param></function>` |
| Gemma native | the model's own envelope, terminated by `<end_function_call>` |
| FunctionGemma | ditto, different envelope |
| toolcall dict as text | `{"name": "read", "arguments": {"path": "x"}}` |

Stop token is `</function>` when tools are bound (single path — the
gemma270 `<end_function_call>` branch was retired with that backend).

`_looks_like_call` (`chat_model.py:220-231`) sniffs for `<|tool_call>`,
`<function`, `<start_function_call>`, `action`, or a paired `"name"`/`"arguments"`
— used to decide whether a retry is warranted.

`_norm_op` (`terminal.py:183-218`) fuzzy-maps the op name. It matches on
op-specific boundaries rather than prefix, which is what stops `python_exec`
and `exec_bg` from collapsing into `exec` — a bug two eval runs found.

### Retry

`_collect_raw` (`chat_model.py:344-360`) makes exactly one retry, with a nudge
string from `_needs_retry` (`chat_model.py:233-248`) appended. `_repeat_note`
(`chat_model.py:390-400+`) detects the last two identical read-only calls and
appends a short-circuit note instead of letting the model loop.

## 2.3 The router

`SemanticRouter` (`src/agent/tool_router.py`) picks which ops to even show the
model, using `BAAI/bge-small-en-v1.5` on CPU (`ROUTER_MODEL_ID`, line 31).

```
text ──▶ embed ──▶ cosine vs each op's blurb ──▶ top-k (k=3)
                                                 │
                            deps + followups ─────┘
                                    │
                            filtered by ALLOWED_OPS
```

Model and tokenizer load once behind a lock (`tool_router.py:78-113`). The
embedding matrix is normalized once (`tool_router.py:106-108`). Routing adds
`_ROUTE_DEPS` and `_ROUTE_FOLLOWUP` expansions (`tool_router.py:147-159`) and
returns `None` on any failure, which degrades to showing every op.

`InjectToolMiddleware` (`tool_router.py:177-206`) publishes the selection
through a `ContextVar`, `current_ops()` (`tool_router.py:63-66`), which
`LocalChatModel._prompt_tools` (`chat_model.py:107-138`) reads. On native-spec
backends (`bonsai`, `qwen`) it skips the embedding router entirely
(`chat_model.py:112`) and uses the keyword shortlist instead.

## 2.4 Policy

Two gates, both in `src/tools/terminal.py`, checked before anything runs.

### Deny — never runs, no confirmation can unlock it

`_DENY_RES` (`terminal.py:114-126`):

```
fork bomb              rm -rf /   rm -rf ~/   mkfs   dd of=/dev/
shutdown/poweroff/reboot/halt      : > /dev/sd   chmod -R 777 /
mv … /dev
```

### Confirm — runs only after explicit user yes

`_CONFIRM_RES` (`terminal.py:129-147`):

```
sudo   eval   $(…)   `…`   process substitution
rm  mv  dd  chmod  chown
git push --force      git reset --hard
curl|wget … | sh|bash
ssh   docker|podman   systemctl|service
pip|pip3|npm install|uninstall      apt/apt-get
```

### The confirmation flow

`VoiceAgent._confirm_wrapper` (`src/main.py:534-551`) wraps `confirm_fn`. It
understands three user answers:

| Answer | Effect |
|---|---|
| `true` | run once |
| `false` | cancel → `term/deny` |
| `"always"` / `"a"` / `"allowlist"` | run once **and** `remember_approval` |

Approvals live in `self.approvals`, keyed by session and by the first token of
the command or the op name (`main.py:467-472`); `is_approved` checks it
(`main.py:474-481`). **This is per agent process, in RAM** — restart the bridge
and every allowlist entry is gone.

A blocked or cancelled call comes back as an observation string starting with
`Blocked:` or `Cancelled`, which `_agent_invoke` converts to `term/deny`
(`main.py:731-743`).

## 2.5 MCP

`src/agent/mcp/config.py` resolves servers in this order (`config.py:1-128`):

```
explicit path → $MCP_CONFIG → configs/mcp_servers.yaml → server_connection()
```

The fallback builds a stdio connection to `python -m src.agent.mcp.server`
with cwd at repo root and a merged `PYTHONPATH` (`config.py:45-57`).
`_remote_connection` handles `http(s)` transports (`config.py:86-91`).
`load_connections` returns `(connections, tool_name_prefix)` so multiple
servers can expose the same tool name without collision (`config.py:94-128`).

`load_extra_tools` (`src/main.py:258-263`) merges discovered tools into the
agent alongside the native ops.

## 2.6 Prompt contract

`TERMINAL_PREAMBLE` (`src/agent/prompts.py:12`) is the single shared contract,
reused by the voice/TTS system prompt, the deep-agent system prompt, and
`/browser`. It states the ops in prose, gives XML *and* JSON examples, and
spells out the rules that matter:

- reply with a chat sentence **or** exactly one JSON action, never both
- no plan narration as a reply
- prefer `read`/`grep` before `write`/`edit`
- destructive ops are blocked
- never repeat an identical call whose result is already in the transcript

That last rule exists because the eval runs kept finding models looping on
identical tool calls.

## 2.7 See also

- [01-agent.md](01-agent.md) — who calls these
- [03-memory.md](03-memory.md) — what a call returns into
