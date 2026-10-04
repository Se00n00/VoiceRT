// Bridge client: HTTP + persistent WS to the local agent backend.
// The TUI spawns bridge.py itself (single local app); VOICE_BRIDGE can
// point at an already-running backend instead. No deps (Node globals).
import { b64ToF32 } from "./audio.js";

export let API = process.env.VOICE_BRIDGE ?? "http://127.0.0.1:8004";
export function setAPI(url: string): void {
  API = url.replace(/\/$/, "");
}
function wsURL(): string {
  return API.replace(/^http/, "ws");
}

export type TurnEvent =
  | { event: "action"; action: Record<string, unknown>; ts?: number }
  | { event: "observation"; observation: string; ts?: number }
  // two-brain delegation: which model took the turn, and why
  | { event: "route"; brain: "front" | "worker"; kind: string; reason: string; forced: boolean; ts?: number }
  | { event: "chat"; reply: string; ts?: number }
  | { event: "thinking"; text: string; append?: boolean; ts?: number }
  | { event: "token"; piece: string; ts?: number }
  | { event: "audio"; wav_b64: string; sr: number; ts?: number }
  | { event: "confirm"; action: Record<string, unknown>; ts?: number }
  | { event: "summary"; reply: string; ts?: number }
  // One-shot: the model-named session title, sent after the first turn.
  | { event: "title"; title: string; session_id?: string; ts?: number }
  | { event: "stuck"; reason: string; ts?: number }
  | { event: "error"; message: string; ts?: number }
  // the running turn was aborted (client sent {"type":"cancel"})
  | { event: "cancelled"; ts?: number }
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

export async function say(text: string): Promise<{ kind: string; wav_b64?: string; sr?: number; message?: string }> {  const r = await fetch(`${API}/term/say`, {
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

  /** Abort the running turn; the bridge answers with a `cancelled` event. */
  cancel(): void {
    try {
      this.ws?.send(JSON.stringify({ type: "cancel" }));
    } catch {
      /* dead socket — nothing to cancel */
    }
  }

  close(): void {
    try {
      this.ws?.close();
    } catch {
      /* ignore */
    }
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws !== null;
  }
}

/** Persistent turn socket for the autonomous DeepAgent (MCP + todos).
 * Carries the confirm gate (the permission prompt) and cancel, same
 * protocol as TermSocket. */
export class DeepSocket {
  private ws: WebSocket | null = null;
  onEvent: (e: TurnEvent) => void = () => {};
  onConfirm: (action: Record<string, unknown>) => void = () => {};
  onOpen: () => void = () => {};
  onClose: () => void = () => {};

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${wsURL()}/deep`);
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

  /** Abort the running turn; the bridge answers with a `cancelled` event. */
  cancel(): void {
    try {
      this.ws?.send(JSON.stringify({ type: "cancel" }));
    } catch {
      /* dead socket — nothing to cancel */
    }
  }

  close(): void {
    try {
      this.ws?.close();
    } catch {
      /* ignore */
    }
    this.ws = null;
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
  const root = path.resolve(import.meta.dirname, "..", "..", "..");
  const venvPy = path.join(root, ".venv", "bin", "python");
  const py = fs.existsSync(venvPy) ? venvPy : "python3";
  const port = process.env.VOICE_PORT ?? "8004";
  setAPI(`http://127.0.0.1:${port}`);
  log(`starting local agent…`);
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
    // Ready/missing/error only: dependency chatter (HF Hub auth warnings,
    // rate-limit notices, download progress) must never reach the status line.
    if (!/ready|missing|error/i.test(s)) return;
    if (/huggingface|hf hub|rate.?limit|HF_TOKEN/i.test(s)) return;
    log(`agent: ${s.slice(0, 160)}`);
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

export type ModelInfo = {
  name: string;
  label: string;
  backend: string;
  desc: string;
  ready: boolean;
  current: boolean;
};

export async function getModel(): Promise<{ current: string; available: ModelInfo[] }> {
  const r = await fetch(`${API}/model`);
  return (await r.json()) as { current: string; available: ModelInfo[] };
}

/**
 * Persisted session name, for a resumed session (`voicert -s <id>`).
 * Model-free on the bridge side: it only reads sessions/<id>.json, so
 * quitting a resumed session without speaking still gets its old name.
 */
export async function fetchTitle(sid: string): Promise<string> {
  if (!sid) return "";
  try {
    const r = await fetch(`${API}/term/title?sid=${encodeURIComponent(sid)}`);
    const j = (await r.json()) as { title?: string };
    return String(j.title ?? "");
  } catch {
    return "";
  }
}

export async function switchModel(name: string): Promise<{ kind: string; current?: string; label?: string; message?: string; note?: string; compacted?: string }> {
  const r = await fetch(`${API}/model/switch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return (await r.json()) as { kind: string; current?: string; label?: string; message?: string; note?: string; compacted?: string };
}

// --- dashboard telemetry ------------------------------------------------
// The left panel polls these every second; every one never
// raises into the render path (callers catch).

export type Metrics = {
  uptime_s: number;
  turns: { done: number; total_s: number; last_s: number };
  queue: { pending: number; locks: number };
  vram: { allocated_mb: number; reserved_mb: number; peak_mb: number; total_mb: number; name: string; cuda: boolean };
  ram: { total_mb: number; used_mb: number; available_mb: number };
  cpu: { percent: number; load: number[]; count: number };
  gpu_util: number;
};

export async function getMetrics(): Promise<Metrics> {
  const r = await fetch(`${API}/metrics`);
  return (await r.json()) as Metrics;
}

export async function getMetricsContext(sid: string): Promise<{ tokens: number; ctx: number; pct: number }> {
  const r = await fetch(`${API}/metrics/context?sid=${encodeURIComponent(sid)}`);
  return (await r.json()) as { tokens: number; ctx: number; pct: number };
}

export type Legs = {
  legs: { vad: string; stt: string; llm: string; tts: string };
  model: { name: string; label: string; backend: string };
  policy: { ops: string[]; deny: string[]; confirm: string[] };
};

export async function getLegs(): Promise<Legs> {
  const r = await fetch(`${API}/legs`);
  return (await r.json()) as Legs;
}

/** Interim STT: cumulative PCM -> best-effort transcript. */
export async function sttInterim(pcm: Buffer, sr = 16000): Promise<{ kind: string; text?: string; message?: string }> {
  const r = await fetch(`${API}/term/stt/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pcm_b64: pcm.toString("base64"), sr }),
  });
  return (await r.json()) as { kind: string; text?: string; message?: string };
}

/**
 * Streaming TTS: NDJSON, one audio chunk per sentence.
 * `onChunk` may return a promise; the loop awaits it, so
 * sentence playback runs sequentially while synthesis of the
 * next sentence proceeds server-side.
 */
export async function sayStream(
  text: string,
  onChunk: (wav: Float32Array, sr: number, sentence: string) => void | Promise<void>,
): Promise<void> {
  const r = await fetch(`${API}/term/say/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text.slice(0, 500) }),
  });
  if (!r.ok || !r.body) throw new Error(`say/stream ${r.status}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl = buf.indexOf("\n");
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) {
        const j = JSON.parse(line) as { kind: string; wav_b64?: string; sr?: number; sentence?: string; message?: string };
        if (j.kind === "audio" && j.wav_b64) {
          await onChunk(b64ToF32(j.wav_b64), j.sr ?? 24000, j.sentence ?? "");
        } else if (j.kind === "error") {
          throw new Error(j.message ?? "tts failed");
        }
      }
      nl = buf.indexOf("\n");
    }
  }
}
