"""MCP server configs from YAML (the one sanctioned YAML in new code).

Selection order: explicit path > ``$MCP_CONFIG`` > repo-root
``configs/mcp_servers.yaml`` > bundled voice-tools fallback (file absent).

:func:`load_connections` returns ``(connections, tool_name_prefix)``
ready for ``MultiServerMCPClient``. Secrets never live in the YAML —
keys travel via environment only; per-server ``env:`` merges OVER the
process environment.
"""
import os
import sys
from pathlib import Path

__all__ = [
    "CONFIG_ENV_VAR",
    "default_config_path",
    "load_connections",
    "repo_root",
    "resolve_config_path",
    "server_connection",
]

CONFIG_ENV_VAR = "MCP_CONFIG"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> str:
    return str(repo_root() / "configs" / "mcp_servers.yaml")


def resolve_config_path(explicit=None) -> str | None:
    if explicit:
        return str(explicit)
    env = (os.environ.get(CONFIG_ENV_VAR, "") or "").strip()
    if env:
        return env
    default = default_config_path()
    return default if os.path.isfile(default) else None


def server_connection() -> dict:
    root = repo_root()
    env = dict(os.environ)
    pp = str(root)
    env["PYTHONPATH"] = pp + os.pathsep + env["PYTHONPATH"] \
        if env.get("PYTHONPATH") else pp
    return {
        "command": sys.executable or "python3",
        "args": ["-m", "src.agent.mcp.server"],
        "transport": "stdio",
        "cwd": str(root),
        "env": env,
    }


def _stdio_connection(name: str, spec: dict) -> dict:
    root = repo_root()
    command = spec.get("command") or sys.executable or "python3"
    args = spec.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError(f"mcp server {name!r}: 'args' must be a string list")
    cwd = spec.get("cwd") or "."
    cwd = cwd if os.path.isabs(cwd) else str(root / cwd)
    extra = spec.get("env") or {}
    if not isinstance(extra, dict):
        raise ValueError(f"mcp server {name!r}: 'env' must be a mapping")
    env = dict(os.environ)
    pp = str(root)
    env["PYTHONPATH"] = pp + os.pathsep + env["PYTHONPATH"] \
        if env.get("PYTHONPATH") else pp
    for k, v in extra.items():
        env[str(k)] = str(v)
    return {
        "command": str(command),
        "args": list(args),
        "transport": "stdio",
        "cwd": str(cwd),
        "env": env,
    }


def _remote_connection(name: str, spec: dict) -> dict:
    url = (spec.get("url") or "").strip() if isinstance(spec.get("url"), str) else ""
    if not url:
        raise ValueError(f"mcp server {name!r}: transport "
                         f"{spec.get('transport')!r} needs a 'url'")
    return {"transport": spec.get("transport"), "url": url}


def load_connections(config_path=None) -> tuple:
    path = resolve_config_path(config_path)
    if path is None:
        return {"voice-tools": server_connection()}, False
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"mcp config not found: {path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"mcp config bad YAML ({path}): {exc}")
    except OSError as exc:
        raise ValueError(f"mcp config unreadable ({path}): {exc}")
    if data is None:
        return {}, False
    if not isinstance(data, dict):
        raise ValueError(f"mcp config must be a mapping ({path})")
    prefix = data.get("tool_name_prefix", False)
    servers = data.get("servers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"mcp config 'servers' must be a mapping ({path})")
    connections = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            raise ValueError(f"mcp server {name!r}: entry must be a mapping")
        if not spec.get("enabled", True):
            continue
        transport = (spec.get("transport") or "stdio").strip().lower()
        if transport == "stdio":
            connections[str(name)] = _stdio_connection(str(name), spec)
        else:
            connections[str(name)] = _remote_connection(str(name), spec)
    return connections, bool(prefix)
