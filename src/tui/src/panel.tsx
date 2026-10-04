// Left panel: the dashboard. Ten compact sections, most
// single-row, polled every second from the bridge's
// /metrics + /legs. The equalizer (user voice) lives here
// too — it is the only animated part of the panel.
import { createSignal, onCleanup, onMount } from "solid-js";
import { TextAttributes } from "@opentui/core";
import { useTerminalDimensions } from "@opentui/solid";
import {
  getLegs,
  getMetrics,
  getMetricsContext,
  type Legs,
  type Metrics,
} from "./bridge.js";
import { VERSION } from "./epilogue.js";
import type { Shade, Theme } from "./theme.js";

const NB = 24;
const HALF = 1; // equalizer rows: 2*HALF+1 = 3
const BW = 3;
const GAP = "  ";

export type Todos = { content: string; status: string }[];

export type PanelProps = {
  theme: () => Theme;
  shade: () => Shade | null;
  sid: () => string;
  cwd: () => string;
  mode: () => string;
  micOn: () => boolean;
  // VAD: speech detected in the current mic chunk (local gate).
  speech: () => boolean;
  // Dictation: a voice recording is in progress.
  dictating: () => boolean;
  recSeconds: () => number;
  todos: () => Todos;
  viz: () => { mode: string; bins: number[] };
  tick: () => number;
  busy: () => boolean;
  // Backend warm sequence ("starting…", "still warming…", "ready", errors).
  // Pushed by the app's ensureBackend, rendered here instead of the status
  // line so boot chatter lives in the dashboard, not the chat.
  backendNote: () => string;
};

const avg = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + (b ?? 0), 0) / xs.length : 0;

/** MB -> "2.3G" / "412M". */
function fmtMB(mb: number): string {
  if (!Number.isFinite(mb) || mb <= 0) return "—";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)}G` : `${Math.round(mb)}M`;
}

/** Seconds -> "1:02:03" / "0:03". */
function fmtDur(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

/** "/run/media/se00n00/P/VoiceAgent/voice-pipeline" -> "…/P/VoiceAgent/voice-pipeline". */
function fmtCwd(cwd: string): string {
  const parts = String(cwd || "").split("/").filter(Boolean);
  const roots = parts.slice(-3).join("/");
  if (!roots) return "/";
  return parts.length > 3 ? `…/${roots}` : `/${roots}`;
}

function fmtCtx(tokens: number): string {
  return Number.isFinite(tokens) ? tokens.toLocaleString("en-US") : "0";
}

/** Compact counts for the narrow panel: 1234 -> "1.2k", 16384 -> "16.4k". */
function fmtShort(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 1000) return String(Math.round(n));
  const k = n / 1000;
  const s = k >= 100 ? String(Math.round(k)) : (Math.round(k * 10) / 10).toString();
  return `${s}k`;
}

export function Panel(props: PanelProps) {
  const dims = useTerminalDimensions();
  const termCols = () => dims().width;

  const [metrics, setMetrics] = createSignal<Metrics | null>(null);
  const [context, setContext] = createSignal<{
    tokens: number;
    ctx: number;
    pct: number;
  } | null>(null);
  const [legs, setLegs] = createSignal<Legs | null>(null);

  onMount(() => {
    let stop = false;
    const poll = async () => {
      try {
        const [m, ctx] = await Promise.all([
          getMetrics(),
          getMetricsContext(props.sid()).catch(() => null),
        ]);
        if (stop) return;
        setMetrics(m);
        if (ctx) setContext(ctx);
      } catch {
        /* bridge down — keep the last values */
      }
    };
    const pollLegs = async () => {
      try {
        const l = await getLegs();
        if (!stop) setLegs(l);
      } catch {
        /* ignore */
      }
    };
    void poll();
    void pollLegs();
    const t = setInterval(poll, 1000);
    const lt = setInterval(pollLegs, 15000);
    onCleanup(() => {
      stop = true;
      clearInterval(t);
      clearInterval(lt);
    });
  });

  // --- equalizer (moved from app.tsx) --------------------------------
  // Amplitude-smoothed five-band bars: fast attack, slow release,
  // so bars glide instead of jumping frame to frame. Idle = flat
  // equal stubs — bars only vibe on actual sound.
  let smoothBands = [0.24, 0.32, 0.38, 0.32, 0.24];
  const bandsNow = (): number[] => {
    void props.tick(); // re-run the smoothing pass on the animation clock
    const v = props.viz();
    const bins = v.bins;
    const live = v.mode !== "idle";
    const targets = live
      ? [
          Math.max(avg(bins.slice(0, 5)), 0.12),
          Math.max(avg(bins.slice(5, 10)), 0.12),
          Math.min(1, Math.max(avg(bins.slice(10, 15)) * 1.35, 0.18)),
          Math.max(avg(bins.slice(15, 20)), 0.12),
          Math.max(avg(bins.slice(20, 24)), 0.12),
        ]
      : [0.3, 0.3, 0.3, 0.3, 0.3];
    smoothBands = targets.map((tgt, i) => {
      const cur = smoothBands[i] ?? tgt;
      const rate = tgt > cur ? 0.55 : 0.3;
      return cur + (tgt - cur) * rate;
    });
    return smoothBands;
  };
  const barsWidth = BW * 5 + GAP.length * 4;
  const off = () =>
    Math.max(
      0,
      Math.floor((Math.max(10, Math.floor(termCols() * 0.22) - 2) - barsWidth) / 2),
    );
  const barLine = (k: number, bands: number[]) =>
    bands.map((v) => ((v ?? 0) * HALF * 2 >= k ? "█".repeat(BW) : " ".repeat(BW))).join(GAP);
  const eqRows = (): string[] => {
    const bands = bandsNow();
    const o = off();
    const rows: string[] = [];
    for (let k = HALF * 2; k >= 1; k--) {
      rows.push(" ".repeat(o) + barLine(k, bands));
    }
    return rows;
  };

  // --- rows -----------------------------------------------------------
  const rw = () => Math.max(10, Math.floor(termCols() * 0.22) - 2);
  const pad = (s: string) => (s + " ".repeat(rw())).slice(0, rw());
  const theme = () => props.theme();

  // 1. Mic + User Voice + VAD
  const micLabel = () =>
    props.micOn() ? "[●] MIC ON" : "[○] MIC OFF";
  const vadLabel = () => {
    if (!props.micOn()) return "VAD off";
    return `VAD ${props.speech() ? "speech" : "idle"}`;
  };

  // 10. Dictation Status (on/off) + Recording time
  const dictLabel = () => {
    const s = props.recSeconds();
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(Math.floor(s % 60)).padStart(2, "0");
    return `dict ${props.dictating() ? "ON" : "off"} ${mm}:${ss}`;
  };

  // 4. Context: tokens + % used (Bonsai LM session budget).
  // Pct first: the panel clips from the right on narrow terminals, so the
  // headline number survives even when the totals are cut.
  const ctxLabel = () => {
    const c = context();
    if (!c) return `ctx — / —`;
    return `ctx ${c.pct.toFixed(0)}% ${fmtShort(c.tokens)}/${fmtShort(c.ctx)}`;
  };

  // 5. Todos: active + yet to complete enums
  const todosLabel = () => {
    const list = props.todos();
    if (!list.length) return "todos none";
    const active = list.filter((t) => t.status === "in_progress");
    const left = list.filter((t) => t.status !== "completed");
    return `todos ${active.length} act · ${left.length} left`;
  };

  // 7. LLM Queue
  const queueLabel = () => {
    const q = metrics()?.queue;
    const pend = q ? `${q.pending}` : "—";
    return `queue ${pend} · busy ${props.busy() ? "1" : "0"}`;
  };

  // 8. Settings: Models (all stages: LLM, VAD, STT, TTS), monotonic theme color, auth
  const legsLabel = () => {
    const l = legs();
    if (!l) return `llm ${props.mode() === "voice" ? "warming…" : "—"}`;
    return `llm ${l.legs.llm.split(" ")[0] ?? "—"}`;
  };
  const legDetail = (which: "vad" | "stt" | "tts") => {
    const l = legs();
    if (!l) return `${which} —`;
    const raw = l.legs[which];
    const tag = raw.split(" ")[0] ?? "—";
    return `${which} ${tag}`;
  };
  const shadeLabel = () =>
    `shade ${props.shade() ?? "none"} · ${theme().mode}`;
  // Authorization for the CURRENT dir: the confirm/deny gate that guards
  // tools run from cwd (dir itself is on the `cwd` line above; /settings
  // has the full rule lists). Counts first so narrow panels keep them.
  const authLabel = () => {
    const l = legs();
    if (!l) return `auth confirm-gated`;
    const cn = l.policy.confirm.length;
    const dn = l.policy.deny.length;
    return `auth ✓${cn} ✗${dn} cwd-gated`;
  };

  // Agent backend: warm progress while booting, then a steady ready/dot.
  const agentLabel = () => {
    const n = props.backendNote().trim();
    if (n) return `agent ${n}`;
    return metrics() ? "● agent ready" : "○ agent down";
  };

  // 9. Memory & resources: VRAM, RAM, CPU & GPU usage + load, Uptime, Latency
  const vramLabel = () => {
    const v = metrics()?.vram;
    if (!v || !v.cuda) return `VRAM —`;
    return `VRAM ${fmtMB(v.allocated_mb)}/${fmtMB(v.total_mb)}`;
  };
  const ramLabel = () => {
    const r = metrics()?.ram;
    const c = metrics()?.cpu;
    if (!r) return `RAM —`;
    const cpu = c ? ` CPU ${c.percent.toFixed(0)}%` : "";
    return `RAM ${fmtMB(r.used_mb)}${cpu}`;
  };
  const gpuLabel = () => {
    const m = metrics();
    const c = m?.cpu;
    const loadStr = c?.load && c.load.length ? ` · L${c.load[0]?.toFixed(1)}` : "";
    const util = m ? `GPU ${m.gpu_util}%` : "GPU —";
    return `${util}${loadStr}`;
  };
  const upLatLabel = () => {
    const m = metrics();
    const lat = m?.turns?.last_s ? `${m.turns.last_s.toFixed(1)}s` : "—";
    return `up ${fmtDur(m?.uptime_s ?? 0)} · lat ${lat}`;
  };

  return (
    <>
      <text fg={theme().ink} attributes={TextAttributes.BOLD}>
        {pad("info & control")}
      </text>
      <text
        fg={props.micOn() ? theme().accent : theme().dim}
        attributes={props.micOn() ? TextAttributes.BOLD : TextAttributes.NONE}
      >
        {pad(micLabel())}
      </text>
      {/* 1. user voice: smoothed equalizer rows */}
      {eqRows().map((row) => (
        <text fg={theme().ink} attributes={TextAttributes.BOLD}>
          {pad(row)}
        </text>
      ))}
      <text fg={props.speech() ? theme().accent : theme().dim} attributes={props.speech() ? TextAttributes.BOLD : TextAttributes.NONE}>
        {pad(vadLabel())}
      </text>
      {/* 10. Dictation */}
      <text fg={props.dictating() ? theme().warn : theme().dim} attributes={props.dictating() ? TextAttributes.BOLD : TextAttributes.NONE}>
        {pad(dictLabel())}
      </text>
      {/* 3. Application | Version No. */}
      <text fg={theme().ink}>{pad(`VoiceRT | ${VERSION}`)}</text>
      {/* 2. MODE (of agent) */}
      <text fg={theme().dim}>{pad(`${props.mode()} · ${props.sid().slice(0, 8)}`)}</text>
      {/* 4. Context: tokens + % */}
      <text fg={theme().dim}>{pad(ctxLabel())}</text>
      {/* 5. todos */}
      <text fg={theme().dim}>{pad(todosLabel())}</text>
      {/* 6. current dir (last three roots) */}
      <text fg={theme().dim}>{pad(fmtCwd(props.cwd()))}</text>
      {/* 7. LLM Queue */}
      <text fg={theme().dim}>{pad(queueLabel())}</text>
      {/* 8. Settings */}
      <text fg={theme().dim}>{pad(legsLabel())}</text>
      <text fg={theme().dim}>{pad(legDetail("vad"))}</text>
      <text fg={theme().dim}>{pad(legDetail("stt"))}</text>
      <text fg={theme().dim}>{pad(legDetail("tts"))}</text>
      <text fg={theme().dim}>{pad(shadeLabel())}</text>
      <text fg={theme().dim}>{pad(authLabel())}</text>
      {/* 9. Memory & resources */}
      <text fg={theme().dim}>{pad(vramLabel())}</text>
      <text fg={theme().dim}>{pad(ramLabel())}</text>
      <text fg={theme().dim}>{pad(gpuLabel())}</text>
      <text fg={theme().dim}>{pad(upLatLabel())}</text>
      <text fg={metrics() ? theme().ink : theme().danger}>
        {pad(metrics() ? "● bridge" : "○ bridge down")}
      </text>
      {/* Agent backend warm/ready (was status-line chatter, now dashboard) */}
      <text fg={props.backendNote() ? theme().warn : metrics() ? theme().ink : theme().danger}>
        {pad(agentLabel())}
      </text>
    </>
  );
}
