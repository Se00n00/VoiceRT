"""Canonical computer-control implementations (OS-level GUI + apps).

Single-definition rule (same as :mod:`src.agent.mcp.tools`): these plain
functions are THE implementations — :mod:`src.agent.mcp.server` exposes
them over MCP as thin wrappers. All functions are sync and total (never
raise — errors become strings).

Backends are optional and probed lazily so a headless box still imports
and runs everything else:
- screenshot: PIL ``ImageGrab`` (X11) -> ``grim`` (Wayland) -> ImageMagick
  ``import`` (X11). Needs a display; saves a PNG, returns its path.
- click / type_text / press_key / scroll: ``pyautogui`` (pip) if present,
  else ``xdotool``. Both need a display.
- get_active_window / list_windows: ``xdotool`` / ``wmctrl``.
- open_app: no GUI backend needed (PATH lookup + detached spawn).

Missing pieces return ``error: ...`` strings naming the install
(``pip install pyautogui`` / ``apt install xdotool wmctrl``), never an
exception — the agent loop recovers and the operator knows what to add.
Same trust level as shell execution: for local use only.
"""
import importlib.util as _util
import os as _os
import shutil as _shutil
import subprocess as _sp
import tempfile as _tempfile

__all__ = [
    "screenshot",
    "click",
    "type_text",
    "press_key",
    "scroll",
    "get_active_window",
    "list_windows",
    "open_app",
]

_MISSING_GUI = ("error: no GUI backend (install pyautogui via pip, or "
                "xdotool via apt, and run under a display)")


def _have_display() -> bool:
    """Display present (X11 or Wayland variables set).

    ..
    """
    return bool((_os.environ.get("DISPLAY", "") or "").strip()
                or (_os.environ.get("WAYLAND_DISPLAY", "") or "").strip())


def _to_int(value) -> int | None:
    """int or None (digit check, no coercion tricks).

    ..
    """
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else None


def _run(cmd, timeout: int = 15):
    """Run a CLI backend. Returns (ok, stdout-stripped-or-error).

    The single subprocess boundary: spawn failures become strings.
    ..
    """
    try:
        p = _sp.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, f"not installed: {cmd[0]}"
    except _sp.TimeoutExpired:
        return False, f"timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return False, f"error: {exc}"
    if p.returncode != 0:
        err = (p.stderr or "").strip()[:200]
        return False, f"{cmd[0]} failed (rc={p.returncode}): {err}"
    return True, (p.stdout or "").strip()


def _pyautogui():
    """pyautogui module or None (import guard, the only import try).

    ..
    """
    try:
        import pyautogui as g
        return g
    except Exception:
        return None


# -- screenshot ----------------------------------------------------------

def _screenshot_which() -> str:
    """First usable screenshot backend: pil, grim, import, or "".

    Pure probes (package spec, PATH, display vars) — no attempt made.
    ..
    """
    if _os.environ.get("DISPLAY", "").strip() \
            and _util.find_spec("PIL") is not None:
        return "pil"
    if _os.environ.get("WAYLAND_DISPLAY", "").strip() \
            and _shutil.which("grim"):
        return "grim"
    if _os.environ.get("DISPLAY", "").strip() \
            and _shutil.which("import"):
        return "import"
    return ""


def screenshot(path: str = "") -> str:
    """Capture the screen to a PNG file. Never raises.

    ``path`` empty -> temp file. First usable backend runs; anything
    failing inside returns ``error: ...`` at the boundary.
    """
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        dest = (str(path or "").strip()
                or _os.path.join(_tempfile.gettempdir(),
                                 next(_tempfile._get_candidate_names())
                                 + ".png"))
        backend = _screenshot_which()
        if backend == "pil":
            from PIL import Image, ImageGrab
            ImageGrab.grab().save(dest)
            with Image.open(dest) as im:
                w, h = im.size
            return f"saved {dest} ({w}x{h})"
        if backend == "grim":
            ok, out = _run(["grim", dest])
            return f"saved {dest} (via grim)" if ok \
                else f"error: grim failed ({out})"
        if backend == "import":
            ok, out = _run(["import", "-window", "root", dest])
            return f"saved {dest} (via import)" if ok \
                else f"error: import failed ({out})"
        return ("error: no screenshot backend "
                "(need X11+PIL, grim on Wayland, or ImageMagick)")
    except Exception as exc:
        return f"error: screenshot failed ({exc})"[:300]


# -- pointer / keyboard ----------------------------------------------------

_VALID_BUTTONS = ("left", "middle", "right")


def click(x: int, y: int, button: str = "left") -> str:
    """Click at screen coordinates. Never raises."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        xi, yi = _to_int(x), _to_int(y)
        if xi is None or yi is None:
            return "error: x/y must be integers"
        if xi < 0 or yi < 0:
            return "error: x/y must be >= 0"
        b = (str(button or "left").strip().lower() or "left")
        if b not in _VALID_BUTTONS:
            return f"error: button must be one of {', '.join(_VALID_BUTTONS)}"
        g = _pyautogui()
        if g is not None:
            g.click(xi, yi, button=b)
            return f"clicked ({xi}, {yi}) [{b}]"
        btn_num = {"left": "1", "middle": "2", "right": "3"}[b]
        ok, out = _run(["xdotool", "mousemove", str(xi), str(yi),
                        "click", btn_num])
        if ok:
            return f"clicked ({xi}, {yi}) [{b}]"
        if "not installed" in out:
            return _MISSING_GUI
        return f"error: click failed ({out})"[:300]
    except Exception as exc:
        return f"error: click failed ({exc})"[:300]


def type_text(text: str, interval: float = 0.0) -> str:
    """Type text into the focused window. Never raises."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        src = str(text or "")
        if not src:
            return "error: empty text"
        if len(src) > 2000:
            return "error: text over 2000 chars (split it up)"
        gap = max(0.0, float(interval))
        g = _pyautogui()
        if g is not None:
            g.write(src, interval=gap)
            return f"typed {len(src)} chars"
        ok, out = _run(["xdotool", "type", "--clearmodifiers", "--", src],
                       timeout=30)
        if ok:
            return f"typed {len(src)} chars"
        if "not installed" in out:
            return _MISSING_GUI
        return f"error: type failed ({out})"[:300]
    except Exception as exc:
        return f"error: type failed ({exc})"[:300]


def press_key(key: str) -> str:
    """Press a key or ``+``-separated combo (e.g. ``ctrl+alt+t``)."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        parts = [p.strip().lower() for p in str(key or "").split("+")
                 if p.strip()]
        if not parts:
            return "error: empty key (e.g. 'Enter' or 'ctrl+alt+t')"
        if len(parts) > 4:
            return "error: combo too long (max 4 keys)"
        g = _pyautogui()
        if g is not None:
            if len(parts) == 1:
                g.press(parts[0])
            else:
                g.hotkey(*parts)
            return f"pressed {'+'.join(parts)}"
        ok, out = _run(["xdotool", "key", "+".join(parts)])
        if ok:
            return f"pressed {'+'.join(parts)}"
        if "not installed" in out:
            return _MISSING_GUI
        return f"error: key press failed ({out})"[:300]
    except Exception as exc:
        return f"error: key press failed ({exc})"[:300]


def scroll(clicks: int, x: int = -1, y: int = -1) -> str:
    """Scroll vertically (``clicks`` > 0 up, < 0 down), at x/y if given."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        n = _to_int(clicks)
        if n is None:
            return "error: clicks must be an integer"
        if n == 0:
            return "error: clicks is 0 (nothing to scroll)"
        xi, yi = _to_int(x), _to_int(y)
        g = _pyautogui()
        if g is not None:
            if xi is not None and xi >= 0 and yi is not None and yi >= 0:
                g.scroll(n, x=xi, y=yi)
            else:
                g.scroll(n)
            return f"scrolled {n} clicks"
        # xdotool: buttons 4 (up) / 5 (down), repeated |n| times
        btn = "4" if n > 0 else "5"
        if xi is not None and xi >= 0 and yi is not None and yi >= 0:
            ok, out = _run(["xdotool", "mousemove", str(xi), str(yi)])
            if not ok and "not installed" in out:
                return _MISSING_GUI
        reps = min(abs(n), 20)
        for _ in range(reps):
            ok, out = _run(["xdotool", "click", btn])
            if not ok:
                if "not installed" in out:
                    return _MISSING_GUI
                return f"error: scroll failed ({out})"[:300]
        return f"scrolled {n} clicks"
    except Exception as exc:
        return f"error: scroll failed ({exc})"[:300]


# -- windows / apps ----------------------------------------------------------

def get_active_window() -> str:
    """Return the active window id/pid/name. Never raises."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        ok, wid = _run(["xdotool", "getactivewindow"])
        if not ok:
            if "not installed" in wid:
                return ("error: no window backend "
                        "(apt install xdotool wmctrl)")
            return f"error: active window query failed ({wid})"[:300]
        ok, name = _run(["xdotool", "getwindowname", wid])
        name = name if ok else "?"
        ok, pid = _run(["xdotool", "getwindowpid", wid])
        pid = pid if ok else "?"
        return f"id={wid} pid={pid} name={name}"
    except Exception as exc:
        return f"error: active window query failed ({exc})"[:300]


def list_windows(limit: int = 50) -> str:
    """List visible windows (id + name). Never raises."""
    try:
        if not _have_display():
            return "error: no display (DISPLAY/WAYLAND_DISPLAY unset)"
        lim = _to_int(limit)
        lim = max(1, min(lim, 200)) if lim is not None else 50
        if _shutil.which("wmctrl"):
            ok, out = _run(["wmctrl", "-l"])
            if ok:
                lines = [ln for ln in out.splitlines() if ln.strip()][:lim]
                return "\n".join(lines) or "(no windows)"
        # xdotool fallback: ids, then one name query each (capped)
        ok, out = _run(["xdotool", "search", "--onlyvisible", "--name", ""])
        if not ok:
            if "not installed" in out:
                return ("error: no window backend "
                        "(apt install xdotool wmctrl)")
            return f"error: window list failed ({out})"[:300]
        rows = []
        for wid in out.split()[:lim]:
            ok, name = _run(["xdotool", "getwindowname", wid])
            rows.append(f"{wid} {(name if ok else '?')}")
        return "\n".join(rows) or "(no windows)"
    except Exception as exc:
        return f"error: window list failed ({exc})"[:300]


def open_app(name: str, args: str = "") -> str:
    """Launch an application by executable name. Never raises.

    Looks ``name`` up on PATH (else tries it as a path) and spawns it
    detached (new session, stdio to devnull) so it outlives the call.
    """
    try:
        exe = (str(name or "").strip().split() or [""])[0]
        if not exe:
            return "error: empty app name"
        target = _shutil.which(exe) or exe
        if not (_os.path.isfile(target) and _os.access(target, _os.X_OK)):
            return (f"error: app not found: {exe!r} "
                    f"(not on PATH; pass a full executable path)")
        cmd = [target] + [a for a in str(args or "").split() if a]
        p = _sp.Popen(cmd, stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,
                      stderr=_sp.DEVNULL, start_new_session=True)
        return f"started {exe} (pid {p.pid})"
    except Exception as exc:
        return f"error: launch failed ({exc})"[:300]
