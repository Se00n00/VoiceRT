import { u8ToB64 } from "../audio/dsp.js";

// Bridge HTTP+WS client. Pure web build: no node:child_process, no spawning.
// The Tauri app is a pure client — point it at an already-running bridge.py.
export type TurnEvent =
  | { event: "action"; action: Record<string, unknown>; ts?: number }
  | { event: "observation"; observation: string; ts?: number }
  | { event: "route"; brain: string; kind: string; reason: string; forced: boolean; ts?: number }
  | { event: "chat"; reply: string; ts?: number }
  | { event: "thinking"; text: string; append?: boolean; ts?: number }
  | { event: "token"; piece: string; ts?: number }
  | { event: "audio"; wav_b64: string; sr: number; ts?: number }
  | { event: "confirm"; action: Record<string, unknown>; ts?: number }
  | { event: "summary"; reply: string; ts?: number }
  | { event: "title"; title: string; session_id?: string; ts?: number }
  | { event: "stuck"; reason: string; ts?: number }
  | { event: "error"; message: string; ts?: number }
  | { event: "cancelled"; ts?: number }
  | { event: string; [k: string]: unknown };

export let API =
  (import.meta.env.VITE_VOICE_BRIDGE as string | undefined) ??
  (typeof localStorage !== "undefined" ? localStorage.getItem("voicert.bridge") : null) ??
  "http://127.0.0.1:8004";

export function setAPI(url: string): void {
  API = url.replace(/\/$/, "");
  try {
    localStorage.setItem("voicert.bridge", API);
  } catch {
    /* ignore */
  }
}

function wsURL(): string {
  return API.replace(/^http/, "ws");
}

async function json<T>(r: Response): Promise<T> {
  return (await r.json()) as T;
}

export async function health(): Promise<{ ok: boolean; agent_loaded?: boolean; missing?: string[] }> {
  const r = await fetch(`${API}/health`);
  const j = await json<{ ok?: boolean; agent_loaded?: boolean; missing?: string[] }>(r);
  return { ok: !!j.ok, agent_loaded: j.agent_loaded, missing: j.missing ?? [] };
}

export async function stt(
  pcm: Uint8Array,
  sr = 16000,
  opts?: { timeoutMs?: number },
): Promise<{ kind: string; text?: string; message?: string }> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), opts?.timeoutMs ?? 120000);
  try {
    const r = await fetch(`${API}/term/stt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pcm_b64: u8ToB64(pcm), sr }),
      signal: ctrl.signal,
    });
    return json(r);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      return { kind: "error", message: "timed out (model overloaded?)" };
    }
    throw e;
  } finally {
    window.clearTimeout(timer);
  }
}

/** Live voice pipeline: PCM16 mono frames to `/term/stt-stream`, partials
    and one final transcript back. One socket per utterance. */
export type SttStreamEvents = {
  onPartial?: (text: string) => void;
  onFinal?: (text: string) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
};

export function openSttStream(sr: number, events: SttStreamEvents): {
  sendPcm: (pcm: Uint8Array) => void;
  stop: () => void;
  close: () => void;
} {
  const ws = new WebSocket(`${wsURL()}/term/stt-stream`);
  let open = false;
  let closed = false;
  const queue: Uint8Array[] = [];
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    open = true;
    try {
      ws.send(JSON.stringify({ type: "hello", sr }));
    } catch {
      /* ignore */
    }
    for (const chunk of queue.splice(0)) {
      try {
        ws.send(chunk);
      } catch {
        /* ignore */
      }
    }
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data ?? ""));
      if (msg?.event === "partial" && typeof msg.text === "string") {
        events.onPartial?.(msg.text);
      } else if (msg?.event === "final") {
        events.onFinal?.(typeof msg.text === "string" ? msg.text : "");
      } else if (msg?.event === "error") {
        events.onError?.(String(msg.message ?? "stream error"));
      }
    } catch {
      /* ignore non-JSON frames */
    }
  };
  const finish = () => {
    if (closed) return;
    closed = true;
    try {
      ws.close();
    } catch {
      /* ignore */
    }
    events.onClose?.();
  };
  ws.onerror = () => {
    if (!open) events.onError?.("stream unreachable (bridge down?)");
  };
  ws.onclose = () => finish();
  return {
    sendPcm: (pcm) => {
      if (closed || pcm.byteLength === 0) return;
      if (!open) {
        queue.push(pcm);
        return;
      }
      try {
        ws.send(pcm);
      } catch {
        /* ignore */
      }
    },
    stop: () => {
      if (closed) return;
      try {
        if (open) ws.send(JSON.stringify({ type: "stop" }));
        else closed = true;
      } catch {
        /* ignore */
      }
    },
    close: () => finish(),
  };
}

export async function say(text: string): Promise<{ kind: string; wav_b64?: string; sr?: number; message?: string }> {
  const r = await fetch(`${API}/term/say`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text.slice(0, 500) }),
  });
  return json(r);
}

export type ModelInfo = { name: string; label: string; backend: string; desc: string; ready: boolean; current: boolean };

export async function getModel(): Promise<{ current: string; available: ModelInfo[] }> {
  const r = await fetch(`${API}/model`);
  return json(r);
}

export async function switchModel(name: string): Promise<{ kind: string; current?: string; label?: string; message?: string }> {
  const r = await fetch(`${API}/model/switch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return json(r);
}

export type HistoryMsg = { role: string; content: string };

export async function fetchHistory(sid: string): Promise<{
  session_id: string;
  title: string;
  messages: HistoryMsg[];
}> {
  const r = await fetch(`${API}/term/history?sid=${encodeURIComponent(sid)}`);
  return json(r);
}

export async function fetchTitle(sid: string): Promise<string> {
  if (!sid) return "";
  try {
    const r = await fetch(`${API}/term/title?sid=${encodeURIComponent(sid)}`);
    const j = await json<{ title?: string }>(r);
    return String(j.title ?? "");
  } catch {
    return "";
  }
}

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
  return json<Metrics>(r);
}

export async function getMetricsContext(sid: string): Promise<{ tokens: number; ctx: number; pct: number }> {
  const r = await fetch(`${API}/metrics/context?sid=${encodeURIComponent(sid)}`);
  return json(r);
}

export type Legs = {
  legs: { vad: string; stt: string; llm: string; tts: string };
  model: { name: string; label: string; backend: string };
  policy: { ops: string[]; deny: string[]; confirm: string[] };
};

export async function getLegs(): Promise<Legs> {
  const r = await fetch(`${API}/legs`);
  return json<Legs>(r);
}

/** Persistent turn socket with confirm pump. One instance per app. */
export class TurnSocket {
  private ws: WebSocket | null = null;
  onEvent: (e: TurnEvent) => void = () => {};
  onConfirm: (action: Record<string, unknown>) => void = () => {};
  onOpen: () => void = () => {};
  onClose: () => void = () => {};

  private path: string;

  constructor(path = "/term") {
    this.path = path;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      let ws: WebSocket;
      try {
        ws = new WebSocket(`${wsURL()}${this.path}`);
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)));
        return;
      }
      const to = window.setTimeout(() => reject(new Error("ws open timeout (bridge up on :8004?)")), 8000);
      ws.addEventListener("open", () => {
        window.clearTimeout(to);
        this.ws = ws;
        this.onOpen();
        resolve();
      });
      ws.addEventListener("error", () => {
        window.clearTimeout(to);
        reject(new Error("ws error"));
      });
      ws.addEventListener("message", (ev) => {
        try {
          const m = JSON.parse(String(ev.data)) as TurnEvent;
          if (m.event === "confirm") this.onConfirm((m as { action?: Record<string, unknown> }).action ?? {});
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
      /* server side times out the confirm */
    }
  }

  cancel(): void {
    try {
      this.ws?.send(JSON.stringify({ type: "cancel" }));
    } catch {
      /* dead socket */
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
