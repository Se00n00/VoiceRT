// Bridge client: HTTP + persistent WS to the local agent backend.
// The TUI spawns bridge.py itself (single local app); VOICE_BRIDGE can
// point at an already-running backend instead. No deps (Node globals).
export let API = process.env.VOICE_BRIDGE ?? "http://127.0.0.1:8004";
export function setAPI(url: string): void {
  API = url.replace(/\/$/, "");
}
function wsURL(): string {
  return API.replace(/^http/, "ws");
}

export type TurnEvent =
  | { event: "action"; action: Record<string, unknown> }
  | { event: "observation"; observation: string }
  | { event: "chat"; reply: string }
  | { event: "audio"; wav_b64: string; sr: number }
  | { event: "confirm"; action: Record<string, unknown> }
  | { event: "summary"; reply: string }
  | { event: "error"; message: string }
  | { event: string; [k: string]: unknown };

export async function health(): Promise<{ ok: boolean; missing: string[] }> {
  const r = await fetch(`${API}/health`);
  const j = (await r.json()) as { ok?: boolean; missing?: string[] };
  return { ok: !!j.ok, missing: j.missing ?? [] };
}

export async function stt(pcm: Buffer, sr = 16000): Promise<{ kind: string; text?: string; message?: string }> {
  const r = await fetch(`${API}/term/stt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pcm_b64: pcm.toString("base64"), sr }),
  });
  return (await r.json()) as { kind: string; text?: string; message?: string };
}

export async function say(text: string): Promise<{ kind: string; wav_b64?: string; sr?: number; message?: string }> {
  const r = await fetch(`${API}/term/say`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text.slice(0, 500) }),
  });
  return (await r.json()) as { kind: string; wav_b64?: string; sr?: number; message?: string };
}

/** Persistent turn socket with confirm auto-pump. */
export class TermSocket {
  private ws: WebSocket | null = null;
  onEvent: (e: TurnEvent) => void = () => {};
  onConfirm: (action: Record<string, unknown>) => void = () => {};
  onOpen: () => void = () => {};
  onClose: () => void = () => {};

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${wsURL()}/term`);
      const to = setTimeout(() => reject(new Error("ws open timeout (bridge up on :8004?)")), 8000);
      ws.addEventListener("open", () => {
        clearTimeout(to);
        this.ws = ws;
        this.onOpen();
        resolve();
      });
      ws.addEventListener("error", () => {
        clearTimeout(to);
        reject(new Error("ws error"));
      });
      ws.addEventListener("message", (ev) => {
        try {
          const m = JSON.parse(String((ev as MessageEvent).data)) as TurnEvent;
          if (m.event === "confirm") this.onConfirm((m as { action: Record<string, unknown> }).action ?? {});
          else this.onEvent(m);
        } catch {
          /* ignore */
        }
      });
      ws.addEventListener("close", () => {
        this.ws = null;
        this.onClose();
      });
    });
  }

  turn(text: string, sessionId: string, cwd: string): void {
    try {
      this.ws?.send(JSON.stringify({ type: "turn", text, session_id: sessionId, cwd }));
    } catch {
      this.ws = null;
      this.onClose();
    }
  }

  confirm(ok: boolean): void {
    try {
      this.ws?.send(JSON.stringify({ type: "confirm", ok }));
    } catch {
      /* socket died mid-turn; server side times out the confirm */
    }
  }

  get connected(): boolean {
    return this.ws !== null;
  }
}

// --- local backend lifecycle -------------------------------------------
// Single-app UX: the TUI owns its agent. Reuse a healthy backend if one is
// already up; otherwise spawn bridge.py as our child (killed on exit).
// server.py is never involved.
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

let child: ChildProcess | null = null;

async function loaded(): Promise<boolean> {
  try {
    const r = await fetch(`${API}/health`);
    const j = (await r.json()) as { ok?: boolean; agent_loaded?: boolean };
    return !!j?.ok && !!j?.agent_loaded;
  } catch {
    return false;
  }
}

export async function ensureBackend(log: (m: string) => void): Promise<boolean> {
  if (process.env.VOICE_BRIDGE) {
    setAPI(process.env.VOICE_BRIDGE);
    return loaded();
  }
  if (await loaded()) return true;
  const root = path.resolve(import.meta.dirname, "..", "..");
  const venvPy = path.join(root, ".venv", "bin", "python");
  const py = fs.existsSync(venvPy) ? venvPy : "python3";
  const port = process.env.VOICE_PORT ?? "8004";
  setAPI(`http://127.0.0.1:${port}`);
  log(`starting local agent (${py} bridge.py) — first boot warms legs…`);
  try {
    child = spawn(py, ["bridge.py", "--port", port], {
      cwd: root,
      env: { ...process.env, PYTHONPATH: root },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (e) {
    log(`spawn failed: ${String(e)}`);
    return false;
  }
  child.on("error", (e) => log(`agent process error: ${String(e)}`));
  child.stderr?.on("data", (d: Buffer) => {
    const s = String(d).trim().split("\n").pop() ?? "";
    if (/ready|missing|error|warn/i.test(s)) log(`agent: ${s.slice(0, 160)}`);
  });
  const t0 = Date.now();
  for (let i = 0; i < 72; i++) {
    await new Promise((r) => setTimeout(r, 5000));
    if (child.exitCode !== null && child.exitCode !== 0) {
      log(`agent exited during warm (code ${child.exitCode})`);
      child = null;
      return false;
    }
    if (await loaded()) {
      log("local agent ready");
      return true;
    }
    if (i % 6 === 5) log(`still warming… ${Math.round((Date.now() - t0) / 1000)}s`);
  }
  log("agent warm timed out");
  return false;
}

export function stopBackend(): void {
  try {
    child?.kill("SIGTERM");
  } catch {
    /* ignore */
  }
  child = null;
}
