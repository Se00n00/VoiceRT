"""Persistent shell session + background jobs for the terminal harness.

:class:`PersistentShell` keeps ONE ``bash`` process per turn so ``cd``,
environment variables, and shell functions persist across steps (plain
``subprocess.run`` per step cannot do that). Commands are framed with a
unique sentinel line carrying the exit code. On timeout the session is
respawned so a wedged foreground command cannot pollute later steps.

:class:`JobManager` runs long commands in the background: ``start`` returns
immediately with a job id, ``poll`` returns new output since the last poll
plus a running flag. A reader thread drains each process so full pipes can
never deadlock the child.

No weights, no network. Blocking calls run in threads; wrap with
``asyncio.to_thread`` at the LangGraph boundary.
"""
import os
import select
import shlex
import subprocess
import threading
import time
import uuid

__all__ = ["PersistentShell", "JobManager", "JobInfo"]


def _readline(fobj, deadline: float) -> str:
    """Blocking readline would ignore our deadline on quiet processes.

    select() first so a silent child (sleep, prompts) cannot wedge us.
    Returns '' on timeout/EOF (indistinguishable by design: caller treats
    '' as end-of-stream and respawns / moves on). Leftover bytes past the
    newline are kept per-fd so no output is ever lost on the happy path.
    """
    fd = fobj.fileno()
    stash = _LEFTOVER.setdefault(fd, bytearray())
    with _LEFTOVER_LOCK:
        if b"\n" in stash:
            line, _, rest = bytes(stash).partition(b"\n")
            _LEFTOVER[fd] = bytearray(rest)
            return (line + b"\n").decode(errors="replace")
    while time.time() < deadline:
        try:
            r, _, _ = select.select([fd], [], [], max(0.0, deadline - time.time()))
        except Exception:
            return ""
        if not r:
            return ""
        try:
            chunk = os.read(fd, 65536)
        except Exception:
            return ""
        if not chunk:
            return ""
        with _LEFTOVER_LOCK:
            stash = _LEFTOVER.setdefault(fd, bytearray())
            stash.extend(chunk)
            if b"\n" in stash:
                line, _, rest = bytes(stash).partition(b"\n")
                _LEFTOVER[fd] = bytearray(rest)
                return (line + b"\n").decode(errors="replace")
    return ""


_LEFTOVER: dict[int, bytearray] = {}
_LEFTOVER_LOCK = threading.Lock()


class JobInfo:
    """One background job (handle returned by :meth:`JobManager.start`)."""

    def __init__(self, job_id: str):
        self.job_id = job_id


class _Job:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.buf: list[str] = []
        self.offset = 0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.rc: int | None = None
        t = threading.Thread(target=self._drain, daemon=True)
        t.start()

    def _drain(self):
        try:
            for line in self.proc.stdout:
                with self._lock:
                    self.buf.append(line)
                    if len(self.buf) > 2000:
                        del self.buf[:len(self.buf) - 2000]
        except Exception:
            pass
        finally:
            try:
                self.rc = self.proc.wait(timeout=5)
            except Exception:
                self.rc = self.proc.returncode
            self._done.set()

    def poll(self) -> dict:
        with self._lock:
            new = "".join(self.buf[self.offset:])
            self.offset = len(self.buf)
        running = not self._done.is_set()
        return {"running": running, "rc": self.rc, "output": new}


class JobManager:
    """Start/poll/kill background shell commands."""

    def __init__(self, cwd: str = "."):
        self.cwd = os.path.abspath(cwd or ".")
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._seq = 0

    def start(self, command: str) -> str:
        """Launch; returns job id like ``job-1``. Never raises."""
        with self._lock:
            self._seq += 1
            jid = f"job-{self._seq}"
        try:
            proc = subprocess.Popen(
                command, shell=True, executable="/bin/bash",
                cwd=self.cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as exc:
            # store a pre-failed job so poll reports the error
            proc = None  # type: ignore
            job = _Job.__new__(_Job)
            job.proc, job.buf, job.offset = proc, [f"error: {exc}\n"], 0
            job._lock, job._done, job.rc = threading.Lock(), threading.Event(), 127
            job._done.set()
            with self._lock:
                self._jobs[jid] = job
            return jid
        with self._lock:
            self._jobs[jid] = _Job(proc)
        return jid

    def poll(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"running": False, "rc": None,
                    "output": f"error: unknown job {job_id}"}
        out = job.poll()
        out["job"] = job_id
        return out

    def kill(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.proc is None:
            return False
        try:
            job.proc.kill()
            return True
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            try:
                if job.proc is not None:
                    job.proc.kill()
            except Exception:
                pass


class PersistentShell:
    """One bash process reused across steps in a turn.

    ``exec`` sends the command plus a sentinel echo, then reads until the
    sentinel. ``cd``/``export``/functions persist. Timeout respawns the
    shell so later steps start clean.
    """

    def __init__(self, cwd: str = ".", timeout_s: float = 30.0):
        self.cwd = os.path.abspath(cwd or ".")
        self.timeout_s = float(timeout_s)
        self.jobs = JobManager(cwd=self.cwd)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        old, self._proc = self._proc, None
        if old is not None:
            try:
                _LEFTOVER.pop(old.stdout.fileno(), None)
            except Exception:
                pass
            try:
                old.kill()
            except Exception:
                pass
        self._proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-s"],
            cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _run_once(self, command: str, token: str, deadline: float) -> tuple[str, bool]:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.write(f"echo __VOICE_{token}_$? __END\n")
        try:
            self._proc.stdin.flush()
        except Exception as exc:
            return f"error: shell write failed: {exc}", False
        out: list[str] = []
        marker = f"__VOICE_{token}_"
        stdout = self._proc.stdout if self._proc else None
        while time.time() < deadline:
            line = _readline(stdout, deadline) if stdout else ""
            if line == "":
                break
            if marker in line and "__END" in line:
                try:
                    rc = int(line.split(marker, 1)[1].split()[0])
                except Exception:
                    rc = -1
                return "".join(out), rc
            out.append(line)
            if sum(map(len, out)) > 200000:
                break
        return "".join(out), False

    def exec(self, command: str, timeout_s: float | None = None) -> tuple[int, str]:
        """Run one command; returns (rc, output). Thread-safe. Never raises."""
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        token = uuid.uuid4().hex[:8]
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._spawn()
                out, rc = self._run_once(command, token, time.time() + timeout)
            except Exception as exc:
                return 127, f"error: shell failed: {exc}"
            if rc is False:
                # timeout or EOF: respawn so the next step is clean
                try:
                    self._spawn()
                except Exception:
                    pass
                return 124, (out + "\n…[timeout, shell respawned]").strip() or "timeout"
        try:
            wd = self.pwd()
            if wd:
                self.cwd = wd
        except Exception:
            pass
        return rc, out.strip() or "(no output)"

    def pwd(self) -> str:
        """Current directory of the session (for harness cwd tracking)."""
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    return self.cwd
                token = uuid.uuid4().hex[:8]
                self._proc.stdin.write(f"echo __PWD_{token}__; pwd; echo __PWD_{token}__END\n")
                self._proc.stdin.flush()
                marker = f"__PWD_{token}__"
                deadline = time.time() + 5.0
                capture: list[str] = []
                while time.time() < deadline:
                    line = self._proc.stdout.readline() if self._proc.stdout else ""
                    if line == "":
                        break
                    if marker in line and "__END" in line:
                        break
                    capture.append(line)
                # pwd prints between the two markers; take the last line
                for line in reversed(capture):
                    s = line.strip()
                    if s and marker not in s:
                        return s
            except Exception:
                pass
            return self.cwd

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            self.jobs.close()
        except Exception:
            pass

    # context-manager sugar for tests / harness turns
    def __enter__(self) -> "PersistentShell":
        return self

    def __exit__(self, *_) -> None:
        self.close()
