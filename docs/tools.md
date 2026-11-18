# Tools

Every tool the long-running agent can call: parameters, tag groups, who sees
what, and how a new one is added. Source of truth is
`src/agent/mcp/specs.py`; the same functions are served over MCP by
`src/agent/mcp/mcp.py`.

The functions are sync and total: they never raise, and failures come back as
an `error: ...` string.

## Sources

| Source | Tools | Where |
|---|---|---|
| MCP registry | 28 ops in four categories | `src/agent/mcp/specs.py` |
| deepagents built-ins | `write_todos`, `task`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | `src/agent/long_running/builder.py` middleware |
| sentinel | `done` — "I am finished talking", not a callable tool | `src/agent/tool_router.py` |

`ALLOWED_OPS` is `MCP_TOOL_NAMES + ("done", "write_todos")`.
`MCP_TOOL_DEFS` / `MCP_TOOL_NAMES` derive each JSON-Schema spec from the
category function's docstring and signature, so the prompt specs and the MCP
server cannot drift.

## ShellUse — 10

| Tool | Params | Notes |
|---|---|---|
| `bash` | `command`, `background?`, `job_id?` | blocking shell by default; `background=true` returns a job id, `job_id` polls it |
| `python_exec` | `code`, `timeout?` (30) | runs in the turn container when sandboxed |
| `read` | `path`, `offset?` (1), `limit?` (100), `max_chars?` (12000) | 1-based line window |
| `write` | `path`, `content`, `make_parents?` (True) | create or overwrite |
| `edit` | `path`, `old_string`, `new_string`, `replace_all?` | exact replace; fails when `old_string` is absent |
| `list` | `path?` (`.`), `show_hidden?`, `limit?` (200) | dirs first |
| `glob` | `pattern`, `path?` (`.`), `limit?` (50) | recursive file-name search |
| `grep` | `pattern`, `path?` (`.`) | ripgrep fast path, fallback scan |
| `move` | `src`, `dst` | parents created |
| `delete` | `path`, `recursive?` | file, symlink, or directory |

## BrowserUse — 7

| Tool | Params | Notes |
|---|---|---|
| `webfetch` | `url`, `max_chars?` (6000) | one http(s) page as stripped text |
| `websearch` | `query`, `count?` (5) | titles, URLs, snippets |
| `open_url` | `url` | OS default browser, detached |
| `browser_click` | `url`, `selector`, `timeout?` (15), `headless?` (True) | Playwright, CSS selector |
| `browser_type` | `url`, `selector`, `text`, `submit?`, `timeout?`, `headless?` | optionally presses Enter |
| `browser_scroll` | `url`, `direction?` (`down`), `pixels?` (800), `timeout?`, `headless?` | reports the page after |
| `download_file` | `url`, `dest`, `max_bytes?` (50 MB), `timeout?` (60) | http(s) to disk |

## ComputerUse — 8

| Tool | Params | Notes |
|---|---|---|
| `screenshot` | `path?` | PNG of the screen |
| `click` | `x`, `y`, `button?` (`left`) | screen coordinates |
| `type_text` | `text`, `interval?` (0.0) | into the focused window |
| `press_key` | `key` | single key or `+` combo (`ctrl+alt+t`) |
| `scroll` | `clicks`, `x?`, `y?` | positive up, negative down |
| `get_active_window` | — | id/pid/name |
| `list_windows` | `limit?` (50) | visible windows |
| `open_app` | `name`, `args?` | executable name |

## PlanningTools — 3

| Tool | Params | Notes |
|---|---|---|
| `task` | `action?` (`list`), `id?`, `command?` | background job launch/poll/list |
| `question` | `action?` (`pending`), `id?`, `question?`, `options?`, `answer?` | clarification queue |
| `skill` | `name?` | load a `SKILL.md`; blank lists installed |

`PlanningTools.task` (background jobs) is not the deepagents `task` handoff;
they share only the name.

### Skill lookup

`skill(name)` scans, in order:

1. `<repo>/skills/<name>/SKILL.md`
2. `$VOICE_WORKDIR/skills/<name>/SKILL.md`
3. `~/.config/opencode/skills/<name>/SKILL.md`

A blank name lists what is installed. Note: `src/setup.py` seeds
`~/.config/voicert/skills`, which the loader above does not scan yet.

## Deepagents built-ins

| Tool | Middleware | Purpose |
|---|---|---|
| `write_todos` | `TodoListMiddleware` | plan state for the turn |
| `task` | `SubAgentMiddleware` | hand a description to `shell-use` / `browser-use` / `computer-use` |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | `FilesystemMiddleware` | deepagents' own filesystem surface |

`glob` and `grep` exist in both the MCP registry and the deepagents
filesystem surface. Subagent stacks get no handoff `task`, so recursion is
impossible.

## Who sees what

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
`READ_ONLY_FS_TOOLS` in `src/agent/long_running/specialists.py`).

## Routing

With `use_tool_router` on, `SemanticRouter` (`src/agent/tool_router.py`)
embeds the request with `BAAI/bge-small-en-v1.5`, ranks the `ROUTING_BLURBS`,
and adds `_ROUTE_DEPS` / `_ROUTE_FOLLOWUP`. The shortlist reaches the model
through `InjectToolMiddleware` and a `ContextVar`; execution always keeps the
full tag-filtered set. No match means show everything.

## MCP config

`src/agent/mcp/config.py` resolves servers in this order:

```
explicit path → $MCP_CONFIG → ~/.config/voicert/mcp.yaml → bundled voice-tools fallback
```

The fallback is a stdio connection to `python -m src.agent.mcp.mcp` with cwd
at the repo root and a merged `PYTHONPATH`. Secrets travel via environment
only; per-server `env:` merges over the process environment.

## Adding a tool

1. Implement it in the matching category module and add the name to its
   `__all__` (`ShellUse`, `BrowserUse`, `ComputerUse`, `PlanningTools`).
2. Add its tags to `TOOL_TAGS` in `src/agent/mcp/specs.py`.
3. Register it with `app.add_tool(...)` in `src/agent/mcp/mcp.py`.
4. Add a `ROUTING_BLURBS` entry in `src/agent/tool_router.py` if the semantic
   router should surface it.

No hand-written JSON-Schema: the spec is derived from the signature.

## See also

- [architecture/02-tools.md](architecture/02-tools.md) — parser, policy gates, MCP details
- [architecture/01-agent.md](architecture/01-agent.md) — planner and specialists
