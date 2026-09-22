"""Docker-backed sandbox: real isolation for terminal tool execution.

:class:`DockerSandbox` implements deepagents' :class:`BaseSandbox` over the
``docker`` CLI (no new dependencies): only ``execute()``,
``upload_files()``, ``download_files()`` and ``id`` are provided — every
file op (ls/read/write/edit/grep/glob) is derived from those by the base
class.

Isolation model (honest version):

- each turn gets its own container: separate pid/network/mount namespaces,
  ``--cap-drop ALL``, no ``--privileged``, memory/cpu/pids limits.
- the turn cwd is bind-mounted at ``/work`` (read-write), so the file
  workflow keeps working on host files. Filesystem *secrecy* is NOT
  provided — containment covers processes, network, devices and resources.
  Path-escape denial still applies on top (defense in depth).
- ``exec``/``exec_bg``/``poll`` run inside the container; ``read``/``write``/
  ``edit``/``grep``/``list`` operate host-side on the same mounted tree
  (identical bytes, no translation needed for results).

Image requirements: ``bash``, ``python3`` (BaseSandbox helper scripts),
GNU coreutils ``timeout``. The default ``python:3.12-slim`` has all three.

Operator setup: the user needs docker daemon access
(``sudo usermod -aG docker $USER`` + re-login). Without it every docker
call fails with a permission error — :meth:`DockerSandbox.ensure_running`
detects that case and says so explicitly.
"""
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from src.agent.shell import JobManager, PersistentShell, _Job

__all__ = ["DockerSandbox", "DockerSandboxError", "SandboxConfig"]


BOX_DIR = "/work"
"""Container path the host turn cwd is mounted at."""


class DockerSandboxError(RuntimeError):
    """Lifecycle failure (no binary, no daemon, no permission, no image)."""


class DockerSandbox(BaseSandbox):
    """One container per turn, driven through the ``docker`` CLI."""

    def __init__(self, host_cwd: str = ".",
                 image: str = "python:3.12-slim",
                 name: str | None = None,
                 network: str = "none",
                 memory: str = "1g",
                 cpus: str = "2.0",
                 pids_limit: int = 256,
                 docker: str = "docker",
                 timeout: int = 120,
                 pull: bool = True):
        self.host_cwd = os.path.abspath(host_cwd or ".")
        self.image = image
        self.container = name or f"voice-sandbox-{uuid.uuid4().hex[:8]}"
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = int(pids_limit)
        self.docker = docker
        self.default_timeout = int(timeout)
        self.pull = bool(pull)
        self._running = False

    # -- identity ----------------------------------------------------
    @property
    def id(self) -> str:
        return self.container

    # -- path mapping --------------------------------------------------
    def to_box(self, host_path: str) -> str | None:
        """Host abspath under the turn cwd -> container path. None on escape."""
        full = os.path.abspath(host_path)
        rel = os.path.relpath(full, self.host_cwd)
        if rel == ".." or rel.startswith(".." + os.sep):
            return None
        return BOX_DIR if rel == "." else BOX_DIR + "/" + rel.replace(os.sep, "/")

    def to_host(self, box_path: str) -> str | None:
        """Container path -> host abspath. None outside /work."""
        p = str(box_path or "").strip()
        if p == BOX_DIR:
            return self.host_cwd
        if p.startswith(BOX_DIR + "/"):
            return os.path.join(self.host_cwd, p[len(BOX_DIR) + 1:])
        return None

    # -- lifecycle -----------------------------------------------------
    def _cli(self, *args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [self.docker, *args], capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            raise DockerSandboxError(
                f"docker CLI not found ({self.docker!r}). Install Docker first.")

    def _require_daemon(self, res: subprocess.CompletedProcess, what: str):
        err = (res.stderr or "") + (res.stdout or "")
        if res.returncode != 0 and ("permission denied" in err.lower()
                                    or "cannot connect to the docker daemon" in err.lower()
                                    or "is the docker daemon running" in err.lower()):
            if "permission denied" in err.lower():
                raise DockerSandboxError(
                    "Docker daemon denied access. Add your user to the docker "
                    "group (sudo usermod -aG docker $USER) and re-login, then retry.")
            raise DockerSandboxError(
                f"Docker daemon unreachable during {what}: {err.strip()[:300]}")

    def ensure_running(self) -> str:
        """Pull (if missing) + start the turn container. Idempotent."""
        if self._running:
            return self.container
        if shutil.which(self.docker) is None:
            raise DockerSandboxError(
                f"docker CLI not found ({self.docker!r}). Install Docker first.")
        if self.pull:
            probe = self._cli("images", "-q", self.image)
            self._require_daemon(probe, "image probe")
            if probe.returncode == 0 and not (probe.stdout or "").strip():
                pull = self._cli("pull", self.image, timeout=600)
                self._require_daemon(pull, "image pull")
                if pull.returncode != 0:
                    raise DockerSandboxError(
                        f"docker pull {self.image} failed: "
                        f"{(pull.stderr or pull.stdout or '').strip()[:300]}")
        run = self._cli(
            "run", "-d", "--rm",
            "--name", self.container,
            "--network", self.network,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "-v", f"{self.host_cwd}:{BOX_DIR}",
            "-w", BOX_DIR,
            self.image, "sleep", "infinity",
            timeout=120)
        self._require_daemon(run, "container start")
        if run.returncode != 0:
            raise DockerSandboxError(
                f"docker run failed: {(run.stderr or run.stdout or '').strip()[:300]}")
        self._running = True
        return self.container

    def close(self) -> None:
        """Stop + remove the turn container. Never raises."""
        if not self._running:
            return
        self._running = False
        try:
            self._cli("rm", "-f", self.container, timeout=60)
        except Exception:
            pass

    def __enter__(self) -> "DockerSandbox":
        self.ensure_running()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -- BaseSandbox abstract API --------------------------------------
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """One-shot command inside the container (bash). Never raises."""
        t = int(timeout) if timeout is not None else self.default_timeout
        if not command or not str(command).strip():
            return ExecuteResponse(output="", exit_code=1)
        # Container-side kill: killing the docker client alone would orphan
        # the command inside the container, so wrap with GNU timeout (the
        # image contract guarantees coreutils). Exit 124 on timeout.
        inner = f"timeout -s KILL {t} bash -c {shlex.quote(str(command))}"
        try:
            res = subprocess.run(
                [self.docker, "exec", "-w", BOX_DIR, self.container,
                 "bash", "-c", inner],
                capture_output=True, text=True, timeout=t + 15,
                stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            return ExecuteResponse(output=f"docker CLI not found ({self.docker!r}).", exit_code=127)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"timeout after {t}s (docker client).", exit_code=124,
                                   truncated=False)
        except Exception as exc:
            return ExecuteResponse(output=f"error: {exc}", exit_code=1)
        out = (res.stdout or "") + (res.stderr or "")
        if len(out) > 200000:
            out = out[:200000] + "\n…[truncated]"
            return ExecuteResponse(output=out, exit_code=res.returncode, truncated=True)
        return ExecuteResponse(output=out, exit_code=res.returncode)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Copy (container-abs-path, bytes) pairs into the container."""
        out: list[FileUploadResponse] = []
        for path, content in files or []:
            try:
                parent = os.path.dirname(path) or "/"
                mk = self._cli("exec", self.container, "mkdir", "-p", parent, timeout=60)
                if mk.returncode != 0:
                    out.append(FileUploadResponse(
                        path=path, error=f"mkdir failed: {(mk.stderr or '').strip()[:200]}"))
                    continue
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(content)
                    tmp = f.name
                try:
                    cp = self._cli("cp", tmp, f"{self.container}:{path}", timeout=120)
                finally:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                if cp.returncode != 0:
                    out.append(FileUploadResponse(
                        path=path, error=f"docker cp failed: {(cp.stderr or '').strip()[:200]}"))
                else:
                    out.append(FileUploadResponse(path=path, error=None))
            except DockerSandboxError as exc:
                out.append(FileUploadResponse(path=path, error=str(exc)[:200]))
            except Exception as exc:
                out.append(FileUploadResponse(path=path, error=f"error: {exc}"[:200]))
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Copy container-abs-paths out as bytes."""
        out: list[FileDownloadResponse] = []
        for path in paths or []:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    tmp = f.name
                try:
                    cp = self._cli("cp", f"{self.container}:{path}", tmp, timeout=120)
                    if cp.returncode != 0:
                        out.append(FileDownloadResponse(
                            path=path, content=None,
                            error=f"docker cp failed: {(cp.stderr or '').strip()[:200]}"))
                        continue
                    with open(tmp, "rb") as f:
                        data = f.read()
                finally:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                out.append(FileDownloadResponse(path=path, content=data, error=None))
            except DockerSandboxError as exc:
                out.append(FileDownloadResponse(path=path, content=None, error=str(exc)[:200]))
            except Exception as exc:
                out.append(FileDownloadResponse(path=path, content=None, error=f"error: {exc}"[:200]))
        return out


class SandboxConfig:
    """Turn-container knobs. ``None`` (default) = host execution as before."""

    def __init__(self, image: str = "python:3.12-slim",
                 network: str = "none",
                 memory: str = "1g",
                 cpus: str = "2.0",
                 pids_limit: int = 256,
                 pull: bool = True):
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = int(pids_limit)
        self.pull = bool(pull)

    def create(self, host_cwd: str = ".") -> DockerSandbox:
        return DockerSandbox(
            host_cwd=host_cwd, image=self.image, network=self.network,
            memory=self.memory, cpus=self.cpus, pids_limit=self.pids_limit,
            pull=self.pull)


class DockerShell(PersistentShell):
    """Persistent bash INSIDE the turn container (same sentinel protocol).

    Transport swap only: stdio pipes go to ``docker exec -i`` instead of a
    local bash, so ``cd``/env persist in the container. ``cwd`` stays a
    HOST path (translated from the container ``pwd``) so host-side file
    ops keep working on the bind-mounted tree.
    """

    def __init__(self, sandbox: DockerSandbox, timeout_s: float = 30.0):
        self.sandbox = sandbox
        self.box_cwd = BOX_DIR
        super().__init__(cwd=sandbox.host_cwd, timeout_s=timeout_s)

    def _make_jobs(self):
        return DockerJobs(self)

    def _spawn_argv(self):
        return (["docker", "exec", "-i", self.sandbox.container,
                 "bash", "--noprofile", "--norc", "-s"], None)

    def pwd(self) -> str:
        raw = super().pwd()
        host = self.sandbox.to_host(raw)
        if host is not None:
            self.box_cwd = raw
            return host
        # shell wandered outside /work (no host meaning): keep last host cwd
        return self.cwd


class DockerJobs(JobManager):
    """Background jobs inside the turn container.

    The host-side reader thread drains the ``docker exec`` client stdout,
    so :meth:`poll` semantics match :class:`JobManager`. Killing the job
    kills the client; the turn-end container removal reaps stragglers.
    """

    def __init__(self, shell: DockerShell):
        super().__init__(cwd=shell.cwd)
        self._shell = shell

    def start(self, command: str) -> str:
        """Launch inside the container at the session's box cwd."""
        import threading as _th

        with self._lock:
            self._seq += 1
            jid = f"job-{self._seq}"
        box = getattr(self._shell, "box_cwd", BOX_DIR) or BOX_DIR
        sb = self._shell.sandbox
        try:
            proc = subprocess.Popen(
                [sb.docker, "exec", "-w", box, sb.container,
                 "bash", "-c", command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                stdin=subprocess.DEVNULL)
        except Exception as exc:
            job = _Job.__new__(_Job)
            job.proc, job.buf, job.offset = None, [f"error: {exc}\n"], 0
            job._lock, job._done, job.rc = _th.Lock(), _th.Event(), 127
            job._done.set()
            with self._lock:
                self._jobs[jid] = job
            return jid
        with self._lock:
            self._jobs[jid] = _Job(proc)
        return jid


__all__ = ["DockerSandbox", "DockerSandboxError", "DockerShell", "DockerJobs",
           "SandboxConfig", "BOX_DIR"]
