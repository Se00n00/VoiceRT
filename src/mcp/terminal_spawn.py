"""Spawn a new OS terminal window (Hyprland-aware). Shared by MCP and TUI.

Fire-and-forget (Option A): a new window appears, the agent keeps working
in the bridged shell's cwd. No PTY plumbing — `tmux capture-pane` can be
added later for Option B without changing this API.
"""
import os
import shutil
import subprocess


def _bin(name: str) -> str | None:
    return shutil.which(name)


def spawn_terminal_window(command: str = "", cwd: str | None = None, title: str = "") -> dict:
    """Spawn a new terminal. Returns {ok, terminal_id, cwd, command} or {ok:False, error}.

    Priority: kitty --detach > hyprctl dispatch > tmux new-window.
    `command` may be "" (empty shell) or e.g. "opencode".
    """
    cwd = os.path.abspath(cwd or os.getcwd())
    cmd = (command or "").strip()
    # Resolve opencode-like short names to the known install locations
    if cmd and not os.path.isabs(cmd) and "/" not in cmd:
        # Bare binary name — let the shell resolve it; but prefer the user-local install
        for cand in (f"/home/se00n00/.opencode/bin/{cmd}", f"/usr/bin/{cmd}"):
            if os.path.exists(cand):
                cmd = cand
                break
    # Kitty is the configured terminal on this box (Hyprland efb5…, kitty 0.48)
    kitty = _bin("kitty")
    if kitty:
        args = ["--detach", "--directory", cwd]
        if title:
            args += ["--title", title]
        args += ["-e", cmd] if cmd else []
        # kitty --detach keeps the window alive after the parent exits
        try:
            p = subprocess.Popen([kitty] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 cwd=cwd, start_new_session=True)
            # Don't wait — fire-and-forget
            return {"ok": True, "terminal_id": f"kitty-{p.pid}", "cwd": cwd, "command": cmd or "(shell)"}
        except Exception as e:
            # fall through to hyprctl/tmux
            last_err = str(e)
        else:
            last_err = ""
        # Hyprland-native fallback (works even when kitty --detach is blocked)
        hypr = _bin("hyprctl")
        if hypr and kitty:
            hy = f"kitty --directory {cwd} -e {cmd}" if cmd else f"kitty --directory {cwd}"
            if title:
                hy = f"kitty --title {title} --directory {cwd} -e {cmd}" if cmd else f"kitty --title {title} --directory {cwd}"
            try:
                subprocess.Popen([hypr, "dispatch", "exec", hy],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return {"ok": True, "terminal_id": f"hypr-{cwd}", "cwd": cwd, "command": cmd or "(shell)"}
            except Exception as e:
                last_err = str(e)
    # tmux fallback — works headless / without a GUI terminal
    tmux = _bin("tmux")
    if tmux:
        tcmd = cmd or "bash"
        try:
            # new-window -c sets cwd; -n sets window name
            wname = (title or cmd or "shell")[:20]
            subprocess.Popen([tmux, "new-window", "-c", cwd, "-n", wname, tcmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return {"ok": True, "terminal_id": f"tmux:{wname}", "cwd": cwd, "command": cmd or "(shell)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "no terminal found (tried kitty, hyprctl, tmux)"}
