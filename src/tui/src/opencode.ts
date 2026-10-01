import { spawn } from "node:child_process";
import fs from "node:fs";

const OPENCODE_BIN = process.env.OPENCODE_BIN || "opencode";

/** Spawn `opencode` in a *new* terminal window, Hyprland-aware. */
export async function spawnOpencode(cwd: string, log: (m: string) => void): Promise<boolean> {
  // Resolve binary (prefer user-local, then system)
  const candidates = [OPENCODE_BIN, "/home/se00n00/.opencode/bin/opencode", "/usr/bin/opencode"];
  let bin = OPENCODE_BIN;
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) { bin = c; break; }
    } catch {}
  }
  // Hyprland+kitty is the actual desktop (verified: Hyprland efb5..., kitty present)
  const kitty = "/usr/bin/kitty";
  const hasKitty = (() => { try { return fs.existsSync(kitty); } catch { return false; } })();

  const trySpawn = (cmd: string, args: string[]): Promise<boolean> =>
    new Promise((res) => {
      const p = spawn(cmd, args, { detached: true, stdio: "ignore", cwd });
      p.unref();
      p.on("error", () => res(false));
      // If the parent exits immediately it likely failed to exec; give it 300ms
      setTimeout(() => res(true), 300);
    });

  // 1) kitty is the configured terminal (ghostty/xterm not installed, tmux is fallback)
  if (hasKitty) {
    // --detach keeps it alive after TUI exits; --directory sets cwd
    if (await trySpawn(kitty, ["--detach", "--directory", cwd, "-e", bin])) {
      log(`spawned opencode in kitty (${bin})`);
      return true;
    }
    // Fallback: hyprctl dispatch exec (Hyprland-native)
    if (await trySpawn("hyprctl", ["dispatch", "exec", `kitty --directory ${cwd} -e ${bin}`])) {
      log("spawned opencode via hyprctl");
      return true;
    }
  }
  // 2) tmux — works even without a GUI terminal
  try {
    const p = spawn("tmux", ["new-window", "-c", cwd, bin], { stdio: "ignore", detached: true });
    p.unref();
    log("spawned opencode in tmux new-window");
    return true;
  } catch {}
  log("no terminal found to spawn opencode (tried kitty, hyprctl, tmux)");
  return false;
}
