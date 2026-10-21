"""Turn box: jail the working directory and own the optional container.

One async context manager per turn. The fast path only re-anchors when
the requested directory differs; the sandbox path creates a fresh
container, swaps the backend, reopens MCP into it, and restores both on
exit. Setup failures raise ``TurnBoxError`` so a turn fails loudly.
"""

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.agent.sandbox.docker import DockerSandboxError, SandboxShellBackend
from src.paths import VoicertPaths

if TYPE_CHECKING:
    from src.agent.long_running.agent import LongRunningConfig

__all__ = ["BOX_DENY_PREFIXES", "TurnBox", "TurnBoxError", "jail_turn_cwd"]


BOX_DENY_PREFIXES = (
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/boot",
    "/root",
    "/var",
)
"""Host prefixes a turn cwd may never mount into a container."""


class TurnBoxError(RuntimeError):
    """Turn-box setup failure (rejected cwd, docker down). Never silent."""


def _data_dir() -> str:
    """Local XDG data dir, shared with the rest of the app."""
    return VoicertPaths.data_dir()


def jail_turn_cwd(cwd: str, sessions_dir: str) -> str | None:
    """Requested cwd -> jailed host abspath, None when rejected.

    Rejects filesystem root, system prefixes, the session store, and
    home secret dirs; creates anything else. Pure except the mkdir.
    ..
    """
    raw = os.path.abspath(os.path.expanduser(str(cwd or ".") or "."))
    if raw == os.sep:
        return None
    guarded = BOX_DENY_PREFIXES + (
        os.path.abspath(os.path.expanduser(str(sessions_dir or ""))),
    )
    for prefix in guarded:
        if not prefix or prefix == os.sep:
            continue
        if raw == prefix or raw.startswith(prefix + os.sep):
            return None
    home = os.path.expanduser("~")
    for secret in (".ssh", ".gnupg"):
        g = os.path.join(home, secret)
        if raw == g or raw.startswith(g + os.sep):
            return None
    try:
        os.makedirs(raw, exist_ok=True)
    except Exception:
        return None
    return raw


class TurnBox:
    """Resolve, mount and restore one turn's shell backend."""

    def __init__(
        self,
        config: "LongRunningConfig",
        get_backend,
        set_backend,
        reanchor,
        lock: Any,
    ):
        """Bind live backend accessors and the shared box lock."""
        self.config = config
        self._get_backend = get_backend
        self._set_backend = set_backend
        self._reanchor = reanchor
        self._lock = lock

    def resolve(self, cwd: str) -> str:
        """Requested cwd -> jailed dir, else work_dir. Raises TurnBoxError.

        ..
        """
        sessions_dir = os.path.expanduser(
            str(getattr(self.config, "sessions_dir", "") or _data_dir() + "/sessions")
        )
        hit = jail_turn_cwd(cwd, sessions_dir)
        if hit is not None:
            return hit
        base = os.path.abspath(
            os.path.expanduser(str(getattr(self.config, "work_dir", "") or "."))
        )
        if jail_turn_cwd(base, sessions_dir) is not None:
            return base
        raise TurnBoxError(f"turn cwd rejected ({str(cwd)[:80]!r}); no safe fallback")

    @asynccontextmanager
    async def open(self, cwd: str):
        """Yield (turn_cwd, turn_backend) and restore state on exit.

        ..
        """
        turn_cwd = self.resolve(cwd)
        box_cfg = getattr(self.config, "sandbox", None)
        if (
            box_cfg is None
            and os.environ.get("VOICE_WORKDIR") == turn_cwd
            and not os.environ.get("VOICE_SANDBOX_CONTAINER")
        ):
            yield turn_cwd, self._get_backend()
            return
        async with self._lock:
            prev_backend, box = self._get_backend(), None
            turn_backend = prev_backend
            boxed = False
            try:
                if box_cfg is not None:
                    try:
                        box = box_cfg.create(turn_cwd)
                        box.ensure_running()
                    except DockerSandboxError as exc:
                        raise TurnBoxError(str(exc)[:300])
                    boxed = True
                    os.environ["VOICE_SANDBOX_CONTAINER"] = box.container
                    turn_backend = SandboxShellBackend(
                        box,
                        timeout=int(
                            getattr(self.config, "exec_timeout_s", 0)
                            or getattr(self.config, "tool_timeout_s", 0)
                            or 30
                        ),
                    )
                os.environ["VOICE_WORKDIR"] = turn_cwd
                self._set_backend(turn_backend)
                await self._reanchor()
                yield turn_cwd, turn_backend
            finally:
                if box is not None:
                    box.close()
                os.environ.pop("VOICE_SANDBOX_CONTAINER", None)
                self._set_backend(prev_backend)
                if boxed:
                    await self._reanchor()
