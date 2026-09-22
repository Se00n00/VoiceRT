"""MCP server exposing terminal tools for the DeepAgent.

Runs as a local python process (stdio) and is launched by the DeepAgent
via ``MultiServerMCPClient``. All tools share a per-CWD ``PersistentShell``
so ``cd``/env persist across calls within a session.

Tools:
- exec / exec_bg / poll
- read / write / edit / grep / list
"""
import os

from mcp.server.fastmcp import FastMCP

from src.agent.shell import PersistentShell
from src.tools.terminal import TerminalAction, check_policy

mcp = FastMCP("voice-terminal")

# One shell per cwd (lazy). The DeepAgent will typically stay in one cwd
# per turn, so this covers persistence without leaking across unrelated cwd.
_shells: dict[str, PersistentShell] = {}


def _shell(cwd: str | None = None) -> PersistentShell:
    key = os.path.abspath(cwd or os.getcwd())
    if key not in _shells:
        _shells[key] = PersistentShell(cwd=key)
    sh = _shells[key]
    try:
        cur = sh.cwd
        if cur != key:
            _shells[cur] = _shells.pop(key)
            return sh
    except Exception:
        pass
    return sh


def _is_safe_path(base: str, target: str) -> bool:
    # Allow /tmp and project tree; deny only truly sensitive escapes
    # For deep agent, we need to allow absolute /tmp paths
    if target.startswith("/tmp/") or target == "/tmp":
        return True
    return os.path.abspath(target).startswith(os.path.abspath(base))


def _policy_ok(action: TerminalAction) -> str | None:
    verdict, reason = check_policy(action)
    if verdict == "deny":
        return f"Blocked: {reason}"
    # Autonomous agent: confirm is auto-approved (still log preview)
    return None


@mcp.tool(name="exec")
def exec_tool(command: str, cwd: str = "") -> str:
    """Run a bash shell command (persistent CWD/env). Prefer read-only commands."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="exec", command=command)
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="exec_bg")
def exec_bg_tool(command: str, cwd: str = "") -> str:
    """Start a long-running shell command in the background. Poll it later."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="exec_bg", command=command)
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="poll")
def poll_tool(job_id: str, cwd: str = "") -> str:
    """Poll a background job started by exec_bg."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="poll", command=job_id)
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="read")
def read_tool(path: str, cwd: str = "") -> str:
    """Read a text file. Use for files, not directories. For directories use list. Path is relative to cwd; cwd defaults to current directory."""
    # Allow absolute /tmp paths even if cwd is elsewhere
    cwd_eff = cwd or _shell().cwd
    if path.startswith("/tmp") and not _is_safe_path(cwd_eff, path):
        # Directly allow /tmp absolute
        from src.agent.terminal import run_command

        act = TerminalAction(op="read", path=path)
        err = _policy_ok(act)
        if err:
            return err
        return run_command(act, cwd="/tmp", shell=_shell("/tmp"))
    sh = _shell(cwd or None)
    act = TerminalAction(op="read", path=path)
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="write")
def write_tool(path: str, text: str, cwd: str = "") -> str:
    """Write text to a file below the working directory."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="write", path=path, text=text)
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="edit")
def edit_tool(path: str, anchor: str, text: str, cwd: str = "") -> str:
    """Anchored patch: replace `anchor` verbatim in the file with `text`."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="edit", path=path, anchor=anchor, text=text)
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="grep")
def grep_tool(pattern: str, path: str = "", cwd: str = "") -> str:
    """Grep a pattern across files below the working directory."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="grep", pattern=pattern, path=path or ".")
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


@mcp.tool(name="spawn_terminal")
def spawn_terminal_tool(command: str = "", cwd: str = "", title: str = "") -> str:
    """Spawn a new OS terminal window. Use for opencode/htop/interactive TUIs."""
    from src.mcp.terminal_spawn import spawn_terminal_window

    res = spawn_terminal_window(command=command, cwd=cwd or None, title=title)
    if res.get("ok"):
        return f"spawned terminal {res['terminal_id']} cwd={res['cwd']} cmd={res['command']}"
    return f"error: {res.get('error')}"

@mcp.tool(name="list")
def list_tool(path: str = "", cwd: str = "") -> str:
    """List directory entries. Use for directories, not files. For files use read."""
    sh = _shell(cwd or None)
    act = TerminalAction(op="list", path=path or ".")
    err = _policy_ok(act)
    if err:
        return err
    from src.agent.terminal import run_command

    return run_command(act, cwd=sh.cwd, shell=sh)


if __name__ == "__main__":
    mcp.run()
