# 02 — Tools

What the agent can actually do, how a model's text becomes a call, and the two
policy gates in front of execution.

## 2.1 The tools

The long-running agent's tools come from three places:

| Source | Tools | Where |
|---|---|---|
| MCP registry | 28 ops (four categories) | `src/agent/mcp/specs.py` |
| deepagents built-ins | `write_todos`, `task`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | `src/agent/long_running/builder.py` middleware |
| sentinel | `done` — "I am finished talking", not a tool | `src/agent/tool_router.py` |

`ALLOWED_OPS` (`src/agent/tool_router.py`) is `MCP_TOOL_NAMES` plus `done` and
`write_todos`. `MCP_TOOL_DEFS` / `MCP_TOOL_NAMES` (`src/agent/mcp/specs.py`)
derive each JSON-Schema spec from the category function's docstring and
signature; `src/agent/mcp/mcp.py` registers the same functions over MCP, so the
two cannot drift. `TERMINAL_TOOLS` and `ALL_TOOL_DEFS` are the same list.

### ShellUse — 10

| Tool | Params | Notes |
|---|---|---|
| `bash` | `command`, `background?`, `job_id?` | blocking shell by default; `background=true` returns a job id, `job_id` polls it |
| `python_exec` | `code`, `timeout?` (30) | in the turn container when sandboxed |
| `read` | `path`, `offset?` (1), `limit?` (100), `max_chars?` | 1-based line window |
| `write` | `path`, `content`, `make_parents?` | create or overwrite |
| `edit` | `path`, `old_string`, `new_string`, `replace_all?` | exact replace; fails when `old_string` is absent |
| `list` | `path?` (`.`), `show_hidden?`, `limit?` (200) | dirs first |
| `glob` | `pattern`, `path?` (`.`), `limit?` (50) | recursive file-name search |
| `grep` | `pattern`, `path?` (`.`) | ripgrep fast path, fallback scan |
| `move` | `src`, `dst` | parents created |
| `delete` | `path`, `recursive?` | file, symlink, or directory |

### BrowserUse — 7

| Tool | Params | Notes |
|---|---|---|
| `webfetch` | `url`, `max_chars?` (6000) | one http(s) page as text |
| `websearch` | `query`, `count?` (5) | titles/URLs/snippets |
| `open_url` | `url` | OS default browser |
| `browser_click` | `url`, `selector`, `timeout?`, `headless?` | Playwright, CSS selector |
| `browser_type` | `url`, `selector`, `text`, `submit?`, `timeout?`, `headless?` | optionally presses Enter |
| `browser_scroll` | `url`, `direction?`, `pixels?`, `timeout?`, `headless?` | reports the page after |
| `download_file` | `url`, `dest`, `max_bytes?`, `timeout?` | http(s) to disk |

### ComputerUse — 8

| Tool | Params | Notes |
|---|---|---|
| `screenshot` | `path?` | PNG of the screen |
| `click` | `x`, `y`, `button?` | left/right |
| `type_text` | `text`, `interval?` | into the focused window |
| `press_key` | `key` | single key or `+` combo |
| `scroll` | `clicks`, `x?`, `y?` | positive up, negative down |
| `get_active_window` | — | id/pid/name |
| `list_windows` | `limit?` (50) | visible windows |
| `open_app` | `name`, `args?` | executable name |

### PlanningTools — 3

| Tool | Params | Notes |
|---|---|---|
| `task` | `action?` (`list`), `id?`, `command?` | background job launch/poll/list (not the handoff tool) |
| `question` | `action?` (`pending`), `id?`, `question?`, `options?`, `answer?` | clarification queue |
| `skill` | `name?` | load a `SKILL.md`; blank lists installed |

### Deepagents built-ins

| Tool | Middleware | Purpose |
|---|---|---|
| `write_todos` | `TodoListMiddleware` | plan state for the turn |
| `task` | `SubAgentMiddleware` | hand a description to `shell-use` / `browser-use` / `computer-use` |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | `FilesystemMiddleware` | deepagents' own filesystem surface |

The `SubAgentMiddleware.task` (specialist handoff) and the MCP
`PlanningTools.task` (background jobs) share a name but are different tools;
subagent stacks get no handoff `task`, so recursion is impossible.

### Who sees what

Tag groups live in `TOOL_TAGS` (`src/agent/mcp/specs.py`):

| Role | Tag | Tools |
|---|---|---|
| planner | `Main` | `bash`, `python_exec`, `read`, `write`, `edit`, `list`, `glob`, `grep`, `webfetch`, `websearch`, `question`, `skill` (12) |
| `shell-use` | `ShellUse` | the 10 ShellUse tools plus `task`, `download_file` (12) |
| `browser-use` | `BrowserUse` | the 7 BrowserUse tools |
| `computer-use` | `ComputerUse` | the 8 ComputerUse tools |

The planner binds `build_execution_tools(..., tags={"Main"})` and still gets
the deepagents built-ins on top. Non-shell specialists also get a read-only
filesystem allowlist (`ls`, `read_file`, `glob`, `grep`,
`READ_ONLY_FS_TOOLS` in `src/agent/long_running/specialists.py`). Tools
discovered from configured MCP servers merge into the same pool, then the tag
filter applies.

## 2.2 Parsing a call out of text

`_parse_action` (`src/models/chat_model.py`) splits thinking from the answer
with `LlmModel.split_thinking`, then asks the model leg's
`extract_tool_calls` (`src/models/llm.py`) for calls — first on the answer,
then on the raw stream as a fallback.

The accepted shapes:

| Shape | Example |
|---|---|
| Hermes tool block | `<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>` |
| bare OpenAI dict | `{"name": "read", "arguments": {"path": "x"}}` |

Names pass through verbatim; mapping a name to execution lives outside the
leg. Anything without a call is plain chat.

### Retry

`_collect_raw` (`src/models/chat_model.py`) makes exactly one retry, with a
nudge string from `_needs_retry` appended when the output is degenerate or an
echo of the prompt. `_repeat_note` detects the last two identical read-only
calls and appends a short-circuit note instead of letting the model loop.

## 2.3 The router

`SemanticRouter` (`src/agent/tool_router.py`) picks which ops to even show the
model, using `BAAI/bge-small-en-v1.5` on CPU (`ROUTER_MODEL_ID`).

```
text ──▶ embed ──▶ cosine vs each op's blurb ──▶ top-k (config tool_top_k, 5)
                                                 │
                            deps + followups ─────┘
                                    │
                            filtered by ALLOWED_OPS
```

Model and tokenizer load once behind a lock; the embedding matrix is
normalized once. Routing adds `_ROUTE_DEPS` and `_ROUTE_FOLLOWUP` expansions
and returns `None` on any failure, which degrades to showing every op.

`InjectToolMiddleware` (`src/agent/long_running/middleware/inject.py`)
publishes the selection through a `ContextVar`, `current_ops()`, which
`LocalChatModel._prompt_tools` reads. On native-spec backends (`bonsai`,
`qwen`) it skips the embedding router entirely and uses the keyword shortlist
instead.

## 2.4 Policy

Two gates, both in `src/agent/long_running/middleware/policy.py`, checked before anything runs.

### Deny — never runs, no confirmation can unlock it

`_DENY_RES` (`src/agent/long_running/middleware/policy.py`):

```
fork bomb              rm -rf /   rm -rf ~/   mkfs   dd of=/dev/
shutdown/poweroff/reboot/halt      : > /dev/sd   chmod -R 777 /
mv … /dev
```

### Confirm — runs only after explicit user yes

`_CONFIRM_RES` (`src/agent/long_running/middleware/policy.py`):

```
sudo   eval   $(…)   `…`   process substitution
rm  mv  dd  chmod  chown
git push --force      git reset --hard
curl|wget … | sh|bash
ssh   docker|podman   systemctl|service
pip|pip3|npm install|uninstall      apt/apt-get
```

### The confirmation flow

`TurnRunner._confirm_wrapper` (`src/agent/long_running/turn.py`) wraps `confirm_fn`. It
understands three user answers:

| Answer | Effect |
|---|---|
| `true` | run once |
| `false` | cancel → `term/deny` |
| `"always"` / `"a"` / `"allowlist"` | run once **and** `remember_approval` |

Approvals live in `LongRunningAgent.approvals`, keyed by session and by the
first token of the command or the op name; `is_approved` checks it
(in `LongRunningAgent.approvals`). **This is per agent process, in RAM** — restart the backend
and every allowlist entry is gone.

A blocked or cancelled call comes back as an observation string starting with
`Blocked:` or `Cancelled`, which `_invoke_boxed` converts to `term/deny`
(in `TurnBox.open`).

## 2.5 MCP

`src/agent/mcp/config.py` resolves servers in this order (`config.py:1-128`):

```
explicit path → $MCP_CONFIG → configs/mcp_servers.yaml → server_connection()
```

The fallback builds a stdio connection to `python -m src.agent.mcp.mcp`
with cwd at repo root and a merged `PYTHONPATH` (`config.py:45-57`).
`_remote_connection` handles `http(s)` transports (`config.py:86-91`).
`load_connections` returns `(connections, tool_name_prefix)` so multiple
servers can expose the same tool name without collision (`config.py:94-128`).

`load_extra_tools` (`src/agent/mcp/client.py:38`) merges discovered tools into
the agent alongside the native ops.

## 2.6 Prompt contract

`TERMINAL_PREAMBLE` (`src/agent/prompts/terminal.py`) is the single shared contract,
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
