"""Voice-term TUI — opencode-style full-screen terminal voice agent.

Full alternate-screen app, two clean rectangles::

    ┌ conversation (left) ──────┬─ live audio (right) ┐
    │ user/agent messages,      │  USER  ▂▅▇ live mic │
    │ todos, Cooking… status    │  AGENT ▃▅▂ TTS shape │
    │┌ input ─────────────────┐ │                      │
    ││ chat or /command      │ │                      │
    │└───────────────────────┘ │                      │
    │ Gemma-4 · session · cwd    │                      │
    └───────────────────────────┴──────────────────────┘

Run:  PYTHONPATH=. python tui.py
Keys: Enter send · v voice turn · y/n confirm gate · / commands
      (/new /cwd /clear /voice /help /quit) · q quit

Stop the API server first: the TUI owns the local agent (LLM on CPU/RAM,
STT/TTS on GPU) — two agents side by side fight over the GPU legs.
"""
import argparse
import asyncio
import os
import subprocess
import threading
import uuid

import numpy as np

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, RichLog, Static

NB = 24  # visualizer bins


def spectrum(chunk: np.ndarray, nb: int = NB) -> list[float]:
    """24 log-scaled spectrum bins 0..1 from float32 mono @16k."""
    if chunk.size == 0:
        return [0.0] * nb
    spec = np.abs(np.fft.rfft(chunk * np.hanning(chunk.size)))
    bands = np.array_split(spec, nb)
    vals = [float(np.log1p(b.mean())) for b in bands]
    mx = max(vals) or 1.0
    return [round(v / mx, 2) for v in vals]


class Visualizer(Static):
    """Right rectangle: live USER mic spectrum + AGENT playback shape."""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.user = [0.0] * NB
        self.agent = [0.0] * NB
        self.mode = "idle"  # idle | user | agent
        self._t = 0.0
        self.set_interval(0.1, self._tick)

    def feed(self, mode: str, bins: list[float]) -> None:
        if mode == "user":
            self.user = (list(bins) + [0.0] * NB)[:NB]
        else:
            self.agent = (list(bins) + [0.0] * NB)[:NB]
        self.mode = mode

    def calm(self) -> None:
        self.user = [0.0] * NB
        self.agent = [0.0] * NB
        self.mode = "idle"

    def _tick(self) -> None:
        self._t += 0.1
        self.update(self._frame())

    def _bars(self, bins: list[float], rows: int, color: str, width: int) -> list[str]:
        n = max(8, min(NB, width))  # one cell per bin: never wrap the panel
        bins = (list(bins) + [0.0] * n)[:n]
        out = []
        for r in range(rows, 0, -1):
            out.append("".join(f"[{color}]█[/]" if v * rows >= r else " " for v in bins))
        return out

    def _frame(self):
        from rich.text import Text

        w = max(12, (self.size.width or 30) - 4)
        h = max(8, (self.size.height or 20) - 6)
        half = max(3, h // 2 - 1)
        if self.mode == "idle":
            ub = [0.08 + 0.06 * abs(np.sin(self._t + i * 0.5)) for i in range(NB)]
            ab = [0.08 + 0.06 * abs(np.cos(self._t + i * 0.5)) for i in range(NB)]
        else:
            ub, ab = self.user, self.agent
        lines = ["[bold dim]USER[/]", *self._bars(ub, half, "cyan", w),
                 "", "[bold dim]AGENT[/]", *self._bars(ab, half, "magenta", w)]
        return Text.from_markup("\n".join(lines))


class VoiceTermApp(App):
    """Opencode-style voice terminal: chat left, live audio right."""

    CSS = """
    #left { width: 65%; border: solid #2c3654; }
    #right { width: 35%; border: solid #2c3654; }
    #conv { height: 1fr; }
    #stream { height: auto; max-height: 4; color: #8b93a7; }
    #entry { height: 3; border: solid #38bdf8; }
    #statusbar { height: 1; color: #8b93a7; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit_app", "quit"),
        Binding("v", "voice", "voice"),
        Binding("y", "confirm_yes", "yes", show=False),
        Binding("n", "confirm_no", "no", show=False),
    ]

    def __init__(self, seconds: float = 5.0) -> None:
        super().__init__()
        self.seconds = seconds
        self.agent = None
        self.sid = uuid.uuid4().hex
        self.cwd = os.getcwd()
        self.busy = False
        self.confirm_event: asyncio.Event | None = None
        self.confirm_answer = False

    # -- layout ---------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="left"):
                yield RichLog(id="conv", highlight=True, markup=True, wrap=True)
                yield Static(id="stream")
                yield Input(id="entry", placeholder="Add a follow-up  ( / for commands · v for voice )")
                yield Static(id="statusbar")
            yield Visualizer(id="right")
        yield Footer()

    async def on_mount(self) -> None:
        self.conv = self.query_one("#conv", RichLog)
        self.stream_line = self.query_one("#stream", Static)
        self._stream_buf = ""
        self._stream_tick = 0.0
        self._last_think = ""
        self.entry = self.query_one("#entry", Input)
        self.viz = self.query_one("#right", Visualizer)
        self.query_one("#left").border_title = " conversation "
        self.query_one("#right").border_title = " ● live audio "
        self._status()
        self.run_worker(self._warm(), exclusive=True)

    def _status(self) -> None:
        try:
            model = getattr(getattr(self.agent, "llm", None), "config", None)
            label = getattr(model, "model", None) or "gemma-4-E4B-it"
            # Short label: "google/gemma-4-E4B-it" -> "gemma-4-E4B-it"
            label = str(label).split("/")[-1]
        except Exception:
            label = "gemma-4-E4B-it"
        bar = self.query_one("#statusbar", Static)
        bar.update(f"{label} · session {self.sid[:8]} · "
                   f"{self.cwd} · / commands · v voice")

    # -- model wiring ---------------------------------------------------
    async def _warm(self) -> None:
        """Warm legs one by one with live progress.

        LLM is the Gemma CPU sidecar (llama-server load, no VRAM);
        STT/TTS warm on GPU when CUDA is up. First boot downloads
        weights + GGUF, so it takes a few minutes.
        """
        import time as _time

        from src.main import VoiceAgent

        agent = VoiceAgent()
        t0 = _time.time()
        for name in ("vad", "stt", "llm", "tts"):
            try:
                await getattr(agent, name).warm()
                self.conv.write(f"[green]✓ {name}[/green] "
                                f"[dim]{_time.time() - t0:.0f}s[/dim]")
            except Exception as exc:
                self.conv.write(f"[red]✕ {name}: {exc}[/red] "
                                f"[dim]{_time.time() - t0:.0f}s[/dim]")
        # idempotent second pass: fills missing list + warmed flag
        # (leg backends/tokenizers are cached, so this is fast).
        await agent.warm()
        self.agent = agent
        missing = getattr(agent, "missing", [])
        if missing:
            self.conv.write(f"[yellow]degraded: {missing}[/yellow]")
        else:
            self.conv.write("[green]ready — voice + terminal live.[/green]")
        self._status()
        self.entry.focus()

    async def _confirm(self, action) -> bool:
        op = getattr(action, "op", "?")
        detail = getattr(action, "command", "") or getattr(action, "path", "")
        self.conv.write(f"[bold yellow]⬡ Confirm? [white]{op}[/white] {detail}  (y/n)[/bold yellow]")
        self.confirm_event = asyncio.Event()
        self.confirm_answer = False
        await self.confirm_event.wait()
        self.confirm_event = None
        return self.confirm_answer

    def action_confirm_yes(self) -> None:
        if self.confirm_event is not None:
            self.confirm_answer = True
            self.confirm_event.set()

    def action_confirm_no(self) -> None:
        if self.confirm_event is not None:
            self.confirm_answer = False
            self.confirm_event.set()

    def action_quit_app(self) -> None:
        self.exit()

    # -- input ----------------------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.entry.value = ""
        if not text or self.busy or self.agent is None:
            return
        if text.startswith("/"):
            await self._command(text)
            return
        self.run_worker(self._text_turn(text), exclusive=True)

    async def _command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if cmd in ("/quit", "/q"):
            self.exit()
        elif cmd == "/new":
            self.sid = uuid.uuid4().hex
            self.conv.write(f"[green]new session {self.sid[:8]}[/green]")
            self._status()
        elif cmd == "/clear":
            self.conv.clear()
        elif cmd == "/cwd" and arg and os.path.isdir(arg):
            self.cwd = os.path.abspath(arg)
            self.conv.write(f"[green]cwd {self.cwd}[/green]")
            self._status()
        elif cmd == "/voice":
            self.run_worker(self._voice_turn(), exclusive=True)
        elif cmd == "/help":
            self.conv.write("[dim]/new /cwd PATH /clear /voice [sec] /help /quit · "
                            "v voice · y/n confirm[/dim]")
        else:
            self.conv.write(f"[red]unknown {cmd} — /help[/red]")

    def action_voice(self) -> None:
        if not self.busy and self.agent is not None and self.entry.value.strip() == "":
            self.run_worker(self._voice_turn(), exclusive=True)

    # -- turns ----------------------------------------------------------
    def _stream_clear(self) -> None:
        try:
            self._stream_buf = ""
            self.stream_line.update("")
        except Exception:
            pass

    def _stream_push(self, piece: str) -> None:
        """Accumulate a live token chip into the transient stream line."""
        import time as _time

        try:
            self._stream_buf = (self._stream_buf + str(piece or ""))[-1200:]
            now = _time.monotonic()
            # Throttle widget updates: per-token re-renders flood the loop.
            if now - getattr(self, "_stream_tick", 0.0) < 0.15:
                return
            self._stream_tick = now
            tail = self._stream_buf[-400:]
            self.stream_line.update(f"[dim]{tail}▌[/dim]")
        except Exception:
            pass

    async def _text_turn(self, text: str) -> None:
        self.busy = True
        self.conv.write(f"[bold cyan]› {text}[/bold cyan]")
        self.conv.write("[dim]⬡ Cooking…[/dim]")
        self._stream_clear()
        self._last_think = ""
        loop = asyncio.get_running_loop()
        audio: list[tuple] = []
        saw = {"chat": False}

        async def _run():
            async for event in self.agent.run_text(text, session_id=self.sid, cwd=self.cwd,
                                                   confirm_fn=self._confirm):
                if event.node != "term":
                    continue
                d = event.data or {}
                if event.kind == "action":
                    self._stream_clear()
                    a = d.get("action", {})
                    self.conv.write(f"[dim]▸ {a.get('action')}: "
                                    f"{a.get('command') or a.get('path') or a.get('pattern') or (a.get('code') or '')[:60] or ''}[/dim]")
                elif event.kind == "thinking":
                    # Think traces are first-class: dimmed, never dropped —
                    # but never twice: skip exact repeats of the last trace.
                    think = str(d.get("text", "") or "")
                    if think and think != getattr(self, "_last_think", ""):
                        self._last_think = think
                        self.conv.write(f"[dim italic]› think {think[:1200]}[/dim italic]")
                elif event.kind == "token":
                    self._stream_push(str(d.get("piece", "") or ""))
                elif event.kind == "observation":
                    self._stream_clear()
                    self.conv.write(f"[dim]{str(d.get('observation', ''))[:800]}[/dim]")
                elif event.kind == "confirm":
                    a = d.get("action", {})
                    self.conv.write(f"[yellow]⬡ confirm {a.get('action')}: "
                                    f"{a.get('command') or a.get('path') or ''} "
                                    f"(y/n)[/yellow]")
                elif event.kind == "stuck":
                    self.conv.write(f"[yellow]↻ stuck: {d.get('reason', '')}[/yellow]")
                elif event.kind == "deny":
                    self._stream_clear()
                    self.conv.write(f"[red]✕ {d.get('reason')}[/red]")
                elif event.kind == "queued":
                    self.conv.write(f"[dim]⏳ queued behind the active turn "
                                    f"(#{d.get('position', '?')}) — injected next[/dim]")
                elif event.kind == "chat":
                    self._stream_clear()
                    reply = str(d.get("reply", "") or "")
                    if reply.strip():
                        saw["chat"] = True
                        self.conv.write(f"[white]{reply[:2000]}[/white]")
                elif event.kind == "summary":
                    # Fallback visibility: tools-only turns emit no chat;
                    # surface the summary reply so the turn never looks empty.
                    self._stream_clear()
                    reply = str(d.get("reply", "") or "")
                    if reply.strip() and not saw["chat"]:
                        self.conv.write(f"[white]{reply[:2000]}[/white]")
                elif event.kind == "audio":
                    audio.append((np.asarray(d.get("wav", []), dtype=np.float32),
                                  int(d.get("sr", 24000))))
                elif event.kind == "error":
                    self._stream_clear()
                    self.conv.write(f"[red]error: {d.get('message')}[/red]")

        try:
            await _run()
        finally:
            self.busy = False
        for wav, sr in audio:
            await self._play(wav, sr)

    async def _voice_turn(self) -> None:
        if self.busy or self.agent is None:
            return
        self.busy = True
        loop = asyncio.get_running_loop()
        try:
            self.conv.write("[cyan]recording — speak now[/cyan]")
            wav = await loop.run_in_executor(None, self._record_blocking, self.seconds)
            if wav.size < 1600:
                self.conv.write("[yellow]nothing recorded.[/yellow]")
                return
            segs = await self.agent.vad.segments(wav, 16000)
            if not segs.segments:
                self.conv.write("[yellow]silence — nothing to transcribe.[/yellow]")
                return
            res = await self.agent.stt.transcribe(wav, 16000)
            if not res.text.strip():
                self.conv.write("[yellow]STT empty.[/yellow]")
                return
            self.busy = False
            await self._text_turn(res.text)
        except Exception as exc:
            self.conv.write(f"[red]voice failed: {exc}[/red]")
        finally:
            self.busy = False
            self.viz.calm()

    def _record_blocking(self, seconds: float) -> np.ndarray:
        """arecord capture; spectrum frames posted to the visualizer live."""
        try:
            proc = subprocess.Popen(
                ["arecord", "-q", "-f", "S16_LE", "-r16000", "-c1", "-t", "raw", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.call_from_thread(self.conv.write, "[red]arecord missing.[/red]")
            return np.zeros(0, dtype=np.float32)
        chunks: list[bytes] = []
        stop = threading.Event()

        def _pump():
            while not stop.is_set():
                data = proc.stdout.read(3200) if proc.stdout else b""
                if not data:
                    break
                chunks.append(data)
                f = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
                bins = spectrum(f)
                try:
                    self.call_from_thread(self.viz.feed, "user", bins)
                except Exception:
                    pass

        t = threading.Thread(target=_pump, daemon=True)
        t.start()
        t.join(timeout=seconds)
        stop.set()
        try:
            proc.terminate()
        except Exception:
            pass
        t.join(timeout=2)
        raw = b"".join(chunks)
        if not raw:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0

    async def _play(self, wav: np.ndarray, sr: int) -> None:
        """aplay reply while the AGENT visualizer dances to its envelope."""
        import soundfile as sf
        import tempfile

        if wav.size == 0:
            return
        n = len(wav)
        frames = 60
        env: list[list[float]] = []
        for i in range(frames):
            seg = wav[i * n // frames:(i + 1) * n // frames]
            env.append(spectrum(seg))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, np.asarray(wav, dtype=np.float32), sr)
            proc = await asyncio.create_subprocess_exec(
                "aplay", "-q", path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            dur = n / sr
            for i in range(frames):
                if proc.returncode is not None:
                    break
                self.viz.feed("agent", env[i])
                await asyncio.sleep(max(0.05, dur / frames))
            await proc.wait()
        except FileNotFoundError:
            self.conv.write("[red]aplay missing — text only.[/red]")
        finally:
            self.viz.calm()
            try:
                os.unlink(path)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="voice-term TUI")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    VoiceTermApp(seconds=args.seconds).run()


if __name__ == "__main__":
    main()
