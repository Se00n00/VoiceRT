import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { AskCard, Greeting, Transcript, suggestTypo, type AgentMode, type Msg, type Perf } from "./components/Hero.js";
import { SAMPLES, SAMPLE_CONVERSATION, SAMPLE_NAMES } from "./components/samples.js";
import { DitherBg } from "./components/DitherBg.js";
import { ReactVoiceMode } from "./components/ReactVoiceMode.js";
import { WarpOrb } from "./components/WarpOrb.js";
import collapseSvg from "./assets/collapse.svg?raw";
import expandSvg from "./assets/expand.svg?raw";
import { API, TurnSocket, fetchHistory, fetchTitle, getLegs, getMetrics, getMetricsContext, getModel, health, say, setAPI, stt, switchModel, type Legs, type Metrics, type TurnEvent } from "./api/client.js";
import type { CotStepData } from "./components/chain-of-thought.js";
import { b64ToF32 } from "./audio/dsp.js";
import { createRecorder, playF32, recordSecs, type Recorder } from "./audio/webaudio.js";

const uid = () => Math.random().toString(16).slice(2, 10);
const newSessionId = () => `ses_${uid()}${uid().slice(0, 2)}`;
// Pinned offline sample session: the whole sample conversation, no backend.
const SAMPLE_SID = "sample";
const SAMPLE_NAME = "Sample conversation";

function fmtDur(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s)}s`;
}

type Recent = { id: string; name: string; ts: number };

function loadRecents(): Recent[] {
  const seed: Recent = { id: SAMPLE_SID, name: SAMPLE_NAME, ts: Date.now() };
  try {
    const raw = JSON.parse(localStorage.getItem("voicert.recents") ?? "[]") as Recent[];
    const list = Array.isArray(raw) ? raw.filter((r) => typeof r?.id === "string").slice(0, 20) : [];
    return list.some((r) => r.id === SAMPLE_SID) ? list : [seed, ...list].slice(0, 20);
  } catch {
    return [seed];
  }
}

/** SVG sparkline for a 0..100 series (metrics history). */
function Spark(props: { data: number[]; stroke: string; label: string }) {
  const pts = () => {
    const d = props.data;
    if (d.length === 0) return "";
    return d
      .map((v, i) => {
        const x = d.length === 1 ? 0 : (i / (d.length - 1)) * 100;
        const y = 100 - Math.min(100, Math.max(0, v));
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };
  return (
    <div>
      <div class="mb-1 flex items-center justify-between text-[11px]">
        <span class="lbl">{props.label}</span>
        <span class="val font-mono">{props.data.length > 0 ? Math.round(props.data[props.data.length - 1]!).toFixed(0) : "—"}</span>
      </div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="h-16 w-full">
        <polyline points={pts()} fill="none" stroke={props.stroke} stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" />
      </svg>
    </div>
  );
}

export function App() {
  const [bridgeUrl, setBridgeUrl] = createSignal(API);
  const [bridgeOk, setBridgeOk] = createSignal(false);
  const [agentLoaded, setAgentLoaded] = createSignal(false);
  const [missing, setMissing] = createSignal<string[]>([]);
  const [sessionId, setSessionId] = createSignal(newSessionId());
  const [sessionName, setSessionName] = createSignal("New session");
  const [cwd, setCwd] = createSignal("/");
  const [msgs, setMsgs] = createSignal<Msg[]>([]);
  // Composer text lives here so attachments and /dictate can insert into
  // the island's input from the outside.
  const [input, setInput] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [phase, setPhase] = createSignal("idle");
  const [confirmAction, setConfirmAction] = createSignal<Record<string, unknown> | null>(null);
  const [modelCurrent, setModelCurrent] = createSignal("…");
  const [modelChoices, setModelChoices] = createSignal<{ name: string; label: string; desc: string }[]>([]);
  const [deepSearch, setDeepSearch] = createSignal(false);
  const [think, setThink] = createSignal(false);
  const [codeMode, setCodeMode] = createSignal(false);
  const [autoMode, setAutoMode] = createSignal(false);
  const [agentMode, setAgentMode] = createSignal<AgentMode>("Agent");
  const [perf, setPerf] = createSignal<Perf>("Medium");
  const [showHistory, setShowHistory] = createSignal(true);
  const [recState, setRecState] = createSignal<{ rec: Recorder; levels: number[] } | null>(null);
  const [recSecs, setRecSecs] = createSignal(0);
  // Full voice mode: voice-glow overlay, composer hidden, no dictation.
  const [voiceMode, setVoiceMode] = createSignal(false);
  let recTimer = 0;
  const [metricsOpen, setMetricsOpen] = createSignal(false);
  const [metrics, setMetrics] = createSignal<Metrics | null>(null);
  const [ctxPct, setCtxPct] = createSignal<{ tokens: number; ctx: number; pct: number } | null>(null);
  const [legs, setLegs] = createSignal<Legs | null>(null);
  const [isTauri, setIsTauri] = createSignal(false);
  const [convOpen, setConvOpen] = createSignal(false);
  const [recents, setRecents] = createSignal<Recent[]>(loadRecents());
  const [hist, setHist] = createSignal<{ vram: number; gpu: number; cpu: number; ctx: number }[]>([]);
  const [theme, setTheme] = createSignal<"dark" | "light">(
    (() => {
      try {
        return localStorage.getItem("voicert.theme") === "light" ? "light" : "dark";
      } catch {
        return "dark";
      }
    })(),
  );

  function toggleTheme(): void {
    const next = theme() === "light" ? "dark" : "light";
    setTheme(next);
    try {
      localStorage.setItem("voicert.theme", next);
    } catch {
      /* ignore */
    }
  }

  function touchRecent(id: string, name: string): void {
    setRecents((rs) => {
      const next = [{ id, name, ts: Date.now() }, ...rs.filter((r) => r.id !== id)].slice(0, 20);
      try {
        localStorage.setItem("voicert.recents", JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }

  async function switchRecent(id: string): Promise<void> {
    setConvOpen(false);
    if (id === sessionId()) return;
    setSessionId(id);
    setMsgs([]);
    setBusy(false);
    setConfirmAction(null);
    // Past conversations live in sessions/<sid>.json — restore the whole
    // transcript (user/assistant plus thinking/tool/cot roles).
    try {
      const h = await fetchHistory(id);
      const loaded = historyToMsgs(h.messages);
      if (loaded.length > 0) {
        const name = h.title.trim() || (id === SAMPLE_SID ? SAMPLE_NAME : "Untitled session");
        setSessionName(name);
        setMsgs(loaded);
        touchRecent(id, name);
        return;
      }
    } catch {
      /* backend down — fall through to the offline fallbacks */
    }
    if (id === SAMPLE_SID) {
      setSessionName(SAMPLE_NAME);
      setMsgs(SAMPLE_CONVERSATION.map((m) => ({ ...m, id: uid() })));
      touchRecent(id, SAMPLE_NAME);
      return;
    }
    const t = await fetchTitle(id);
    const name = t.trim() || "Untitled session";
    setSessionName(name);
    touchRecent(id, name);
    push("sys", `switched to ${id}`);
  }

  async function winAction(a: "min" | "max" | "close"): Promise<void> {
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const w = getCurrentWindow();
      if (a === "min") await w.minimize();
      else if (a === "close") await w.close();
      else if (await w.isFullscreen()) await w.setFullscreen(false);
      else await w.toggleMaximize();
    } catch {
      /* browser preview — no window to manage */
    }
  }

  function metricsMini(): string {
    const m = metrics();
    if (!m) return bridgeOk() ? "metrics…" : "bridge down";
    return `VRAM ${(m.vram.allocated_mb / 1024).toFixed(1)}/${(m.vram.total_mb / 1024).toFixed(1)}G · GPU ${Math.round(m.gpu_util)}% · ${fmtDur(m.uptime_s)}`;
  }

  // zoom is disabled app-wide (Tauri zoomHotkeysEnabled=false covers the
  // desktop shell; these cover the browser preview)
  function noZoomWheel(e: WheelEvent): void {
    if (e.ctrlKey) e.preventDefault();
  }
  function noZoomKeys(e: KeyboardEvent): void {
    if ((e.ctrlKey || e.metaKey) && ["+", "-", "=", "0"].includes(e.key)) e.preventDefault();
  }

  const sock = new TurnSocket("/term");
  let pollTimer = 0;

  function push(who: Msg["who"], text: string): void {
    setMsgs((m) => [...m, { id: uid(), who, text }]);
  }

  /** Session-JSON roles -> transcript bubbles (unknown roles read as system lines). */
  function historyToMsgs(hist: { role: string; content: string }[]): Msg[] {
    const out: Msg[] = [];
    for (const m of hist ?? []) {
      const text = String(m?.content ?? "");
      if (!text.trim()) continue;
      const role = String(m?.role ?? "");
      if (role === "user") out.push({ id: uid(), who: "you", text });
      else if (role === "assistant") out.push({ id: uid(), who: "agent", text });
      else if (role === "cot") {
        try {
          const steps = JSON.parse(text) as CotStepData[];
          if (Array.isArray(steps) && steps.length > 0) {
            out.push({ id: uid(), who: "cot", text: "", steps });
            continue;
          }
        } catch {
          /* not steps JSON — show as a system line */
        }
        out.push({ id: uid(), who: "sys", text });
      } else out.push({ id: uid(), who: "sys", text });
    }
    return out;
  }

  function onEvent(e: TurnEvent): void {
    if (e.event === "thinking") {
      setPhase("thinking");
      if (typeof e.text === "string" && e.text.trim()) push("sys", e.text.slice(0, 400));
    } else if (e.event === "token") {
      setPhase("streaming");
    } else if (e.event === "action") {
      const a = (e as { action: Record<string, unknown> }).action ?? {};
      push("sys", `tool: ${String(a.op ?? a.tool ?? "run")} ${JSON.stringify(a).slice(0, 220)}`);
    } else if (e.event === "observation") {
      push("sys", String((e as { observation: string }).observation).slice(0, 800));
    } else if (e.event === "chat" || e.event === "summary") {
      const reply = String((e as { reply: string }).reply ?? "");
      if (reply.trim()) push("agent", reply);
      setPhase("idle");
      setBusy(false);
    } else if (e.event === "title") {
      const t = String((e as { title: string }).title ?? "");
      if (t.trim()) {
        setSessionName(t.trim());
        touchRecent(sessionId(), t.trim());
      }
    } else if (e.event === "stuck") {
      push("err", `stuck: ${String((e as { reason: string }).reason ?? "")}`);
      setBusy(false);
      setPhase("idle");
    } else if (e.event === "error") {
      push("err", String((e as { message: string }).message ?? "error"));
      setBusy(false);
      setPhase("idle");
    } else if (e.event === "cancelled") {
      push("sys", "turn cancelled");
      setBusy(false);
      setPhase("idle");
    }
  }

  async function ensureSocket(): Promise<boolean> {
    if (sock.connected) return true;
    try {
      sock.onEvent = onEvent;
      sock.onConfirm = (a) => {
        if (autoMode()) {
          sock.confirm(true);
          push("sys", `auto-confirmed: ${JSON.stringify(a).slice(0, 160)}`);
        } else {
          setConfirmAction(a);
        }
      };
      sock.onClose = () => setPhase("idle");
      await sock.connect();
      return true;
    } catch {
      return false;
    }
  }

  async function refresh(): Promise<void> {
    try {
      const h = await health();
      setBridgeOk(h.ok);
      setAgentLoaded(!!h.agent_loaded);
      setMissing(h.missing ?? []);
      if (h.ok) {
        try {
          const m = await getModel();
          setModelCurrent(m.current);
          setModelChoices(m.available.map((a) => ({ name: a.name, label: a.label, desc: a.desc })));
        } catch {
          /* model endpoint optional */
        }
        try {
          const met = await getMetrics();
          setMetrics(met);
          try {
            const c = await getMetricsContext(sessionId());
            setCtxPct(c);
            setHist((h) =>
              [
                ...h,
                {
                  vram: met.vram.total_mb > 0 ? (met.vram.allocated_mb / met.vram.total_mb) * 100 : 0,
                  gpu: met.gpu_util,
                  cpu: met.cpu.percent,
                  ctx: c.pct * 100,
                },
              ].slice(-80),
            );
          } catch {
            setCtxPct(null);
          }
        } catch {
          setMetrics(null);
        }
        try {
          setLegs(await getLegs());
        } catch {
          setLegs(null);
        }
      }
    } catch {
      setBridgeOk(false);
      setAgentLoaded(false);
    }
  }

  async function sendText(raw: string): Promise<void> {
    const text = raw.trim();
    if (!text || busy()) return;
    if (text.startsWith("/")) {
      await runCommand(text);
      return;
    }
    if (!(await ensureSocket())) {
      push("err", "bridge is down — start `PYTHONPATH=. .venv/bin/python bridge.py` then retry.");
      return;
    }
    const tagged = withDirectives(text);
    push("you", text);
    setInput("");
    setBusy(true);
    setPhase("working");
    touchRecent(sessionId(), sessionName());
    sock.turn(tagged, sessionId(), cwd());
  }

  /** Toggles + dropdowns become trailing directives on the outgoing turn. */
  function withDirectives(text: string): string {
    const parts: string[] = [];
    if (deepSearch()) parts.push("consult web search tools for current information");
    if (think()) parts.push("think step by step before acting");
    if (codeMode()) parts.push("use code execution tools to verify your work when helpful");
    if (agentMode() === "Assistant") parts.push("answer directly without using tools");
    if (perf() === "High") parts.push("be thorough and detailed");
    else if (perf() === "Low") parts.push("be concise");
    return parts.length > 0 ? `${text} — ${parts.join("; ")}` : text;
  }

  function toggleAuto(): void {
    const next = !autoMode();
    setAutoMode(next);
    // Enabling Auto with a gate pending answers it immediately.
    if (next && confirmAction()) {
      sock.confirm(true);
      setConfirmAction(null);
      push("sys", "auto-confirmed pending action (Auto on)");
    }
  }

  async function attachFile(f: File): Promise<void> {
    try {
      const text = (await f.text()).slice(0, 4000);
      setInput((v) => `${v}${v.trim() ? "\n\n" : ""}[file: ${f.name}]\n${text}`);
      push("sys", `attached ${f.name} (${Math.min(f.size, 4000)} chars into composer)`);
    } catch {
      push("err", `could not read ${f.name}`);
    }
  }

  async function runCommand(line: string): Promise<void> {
    const [cmd, ...rest] = line.split(/\s+/);
    const arg = rest.join(" ").trim();
    setInput("");
    if (cmd === "/new") {
      const sid = newSessionId();
      setSessionId(sid);
      setSessionName("New session");
      setMsgs([]);
      touchRecent(sid, "New session");
      push("sys", `fresh session ${sid}`);
    } else if (cmd === "/clear") {
      setMsgs([]);
    } else if (cmd === "/cwd") {
      if (!arg) push("sys", `cwd: ${cwd()}`);
      else {
        setCwd(arg);
        push("sys", `cwd -> ${arg}`);
      }
    } else if (cmd === "/model") {
      if (!arg) {
        const names = modelChoices().map((c) => c.name).join(", ");
        push("sys", `model: ${modelCurrent()}${names ? `  (available: ${names})` : ""}`);
      } else {
        try {
          const r = await switchModel(arg);
          push(r.kind === "ok" ? "sys" : "err", r.kind === "ok" ? `model -> ${r.current ?? arg}` : (r.message ?? "switch failed"));
          await refresh();
        } catch {
          push("err", "model switch failed (bridge down?)");
        }
      }
    } else if (cmd === "/voice" || cmd === "/dictate" || cmd === "/mic") {
      await recordTurn(cmd === "/mic" ? 5 : Math.min(30, Number(arg) || 5), cmd === "/dictate");
    } else if (cmd === "/sample") {
      const key = (arg || "chain").toLowerCase();
      const thread = SAMPLES[key];
      if (!thread) {
        push("err", `unknown sample ${arg || ""} (available: ${SAMPLE_NAMES.join(", ")})`);
      } else {
        setMsgs((ms) => [...ms, ...thread.map((m) => ({ ...m, id: uid() }))]);
        push("sys", `sample: ${key}`);
      }
    } else if (cmd === "/help") {
      push("sys", "/new /cwd /clear /model /voice /dictate /mic /sample /help — Tab completes, Enter sends what you typed.");
    } else {
      const sug = suggestTypo(cmd ?? "");
      push("err", `unknown command ${cmd ?? ""}${sug ? ` — did you mean ${sug}?` : ""}`);
    }
  }

  async function recordTurn(secs: number, dictateOnly: boolean): Promise<void> {
    try {
      setPhase("recording");
      push("sys", `recording ${secs}s…`);
      const { pcm, sr } = await recordSecs(secs);
      const r = await stt(pcm, sr);
      if ((r.kind !== "ok" && r.kind !== "text") || !r.text?.trim()) {
        push("err", `stt: ${r.message ?? "no speech"}`);
        setPhase("idle");
        return;
      }
      push("sys", `heard: ${r.text}`);
      if (dictateOnly) {
        setInput(r.text);
        setPhase("idle");
      } else {
        setPhase("idle");
        await sendText(r.text);
      }
    } catch (e) {
      push("err", `mic: ${e instanceof Error ? e.message : String(e)}`);
      setPhase("idle");
    }
  }

  /** Inline dictation result: transcribe into the composer (input). */
  async function dictateDone(pcm: Uint8Array): Promise<void> {
    try {
      setPhase("working");
      const r = await stt(pcm, 16000);
      if ((r.kind !== "ok" && r.kind !== "text") || !r.text?.trim()) {
        push("err", `stt: ${r.message ?? "no speech"}`);
      } else {
        setInput((v) => `${v}${v.trim() ? " " : ""}${r.text!.trim()}`);
      }
    } catch (e) {
      push("err", `stt: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPhase("idle");
    }
  }

  /** Start inline recording (equalizer lives in the composer); toggle stops. */
  async function startDictate(): Promise<void> {
    if (recState()) {
      await stopDictate();
      return;
    }
    let rec: Recorder;
    try {
      rec = await createRecorder((rms) => {
        setRecState((s) =>
          s ? { rec: s.rec, levels: [...s.levels.slice(-47), 0.06 + Math.min(1, rms) * 0.9] } : s,
        );
      });
    } catch (e) {
      push("err", `mic: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    setRecState({ rec, levels: new Array(48).fill(0.06) });
    setRecSecs(0);
    recTimer = window.setInterval(() => setRecSecs((s) => s + 1), 1000);
  }

  async function stopDictate(): Promise<void> {
    const s = recState();
    setRecState(null);
    window.clearInterval(recTimer);
    if (!s) return;
    try {
      const { pcm } = await s.rec.stop();
      const secs = pcm.byteLength / 2 / 16000;
      if (secs < 0.5) {
        push("err", `mic captured only ${secs.toFixed(1)}s — check the input device`);
        setPhase("idle");
        return;
      }
      await dictateDone(pcm);
    } catch (e) {
      push("err", `mic: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function cancelDictate(): void {
    const s = recState();
    setRecState(null);
    window.clearInterval(recTimer);
    try {
      s?.rec.cancel();
    } catch {
      /* ignore */
    }
  }

  function attachUrl(url: string): void {
    setInput((v) => `${v}${v.trim() ? "\n\n" : ""}[ref: ${url}]`);
    push("sys", `attached link ${url}`);
  }

  function insertTemplate(text: string): void {
    setInput((v) => `${v}${v.trim() ? "\n\n" : ""}${text}`);
  }

  async function pasteClipboard(): Promise<void> {
    try {
      const items = await navigator.clipboard.read();
      let pasted = 0;
      for (const item of items) {
        for (const type of item.types) {
          if (type.startsWith("image/") || type === "application/octet-stream") {
            const blob = await item.getType(type);
            await attachFile(new File([blob], `clipboard.${type.split("/")[1] ?? "bin"}`, { type }));
            pasted++;
          }
        }
      }
      if (pasted === 0) {
        const text = await navigator.clipboard.readText();
        if (text.trim()) setInput((v) => `${v}${v.trim() ? "\n" : ""}${text.slice(0, 4000)}`);
      }
    } catch {
      /* clipboard blocked — nothing pasted */
    }
  }

  async function speak(text: string): Promise<void> {
    try {
      const r = await say(text);
      if (r.kind === "audio" && r.wav_b64) await playF32(b64ToF32(r.wav_b64), r.sr ?? 24000);
      else push("err", `tts: ${r.message ?? "failed"}`);
    } catch (e) {
      push("err", `tts: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  onMount(() => {
    setIsTauri("__TAURI_INTERNALS__" in window);
    touchRecent(sessionId(), sessionName());
    window.addEventListener("wheel", noZoomWheel, { passive: false });
    window.addEventListener("keydown", noZoomKeys);
    void refresh();
    void fetchTitle(sessionId()).then((t) => {
      if (t.trim()) setSessionName(t.trim());
    });
    pollTimer = window.setInterval(() => void refresh(), 5000);
    void ensureSocket();
  });
  onCleanup(() => {
    window.clearInterval(pollTimer);
    window.clearInterval(recTimer);
    try {
      recState()?.rec.cancel();
    } catch {
      /* ignore */
    }
    window.removeEventListener("wheel", noZoomWheel);
    window.removeEventListener("keydown", noZoomKeys);
    sock.close();
  });
  createEffect(() => {
    document.title = `VoiceRT — ${sessionName()}`;
  });

  // Centered until the first real turn; docked at the bottom afterwards.
  const hasReal = createMemo(() => msgs().some((m) => m.who === "you" || m.who === "agent"));
  const notices = createMemo(() => msgs().filter((m) => m.who === "sys" || m.who === "err"));

  /** Enter full voice mode: glow overlay only — dictation stays off. */
  function enterVoiceMode(): void {
    if (recState()) cancelDictate();
    setVoiceMode(true);
  }

  function composer() {
    return (
      <div class="w-full">
        <AskCard
          value={input()}
          onInput={setInput}
          onSubmit={() => void sendText(input())}
          busy={busy()}
          autoComplete
          recording={recState() !== null}
          recLevels={recState()?.levels ?? []}
          recSecs={recSecs()}
          onMicToggle={() => void startDictate()}
          onVoiceMode={enterVoiceMode}
          onRecCancel={cancelDictate}
          autoMode={autoMode()}
          onToggleAuto={toggleAuto}
          think={think()}
          deepSearch={deepSearch()}
          codeMode={codeMode()}
          showHistory={showHistory()}
          onToggleThink={() => setThink((v) => !v)}
          onToggleDeep={() => setDeepSearch((v) => !v)}
          onToggleCode={() => setCodeMode((v) => !v)}
          onToggleHistory={() => setShowHistory((v) => !v)}
          agentMode={agentMode()}
          onAgentMode={setAgentMode}
          perf={perf()}
          onPerf={setPerf}
          models={modelChoices().map((c) => ({ name: c.name, label: c.label }))}
          modelCurrent={modelCurrent()}
          onModelSelect={(name) => void sendText(`/model ${name}`)}
          onAttach={(f) => void attachFile(f)}
          onAttachUrl={attachUrl}
          onInsertTemplate={insertTemplate}
          onPasteClipboard={() => void pasteClipboard()}
          ctx={ctxPct()}
          sessionLabel={sessionName()}
        />
      </div>
    );
  }

  return (
    <div
      class={`flex h-screen w-screen flex-col overflow-hidden p-[6px] md:p-[4px] ${
        theme() === "light" ? "theme-light bg-[#dfe3ef]" : "bg-black"
      }`}
    >
      <div class="stage flex min-h-0 w-full flex-1 overflow-hidden rounded-[18px]">
        <section class="hero relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl ring-1 ring-black">
        <DitherBg scrollId="hero-scroll" light={theme() === "light"} />

        <div id="hero-scroll" class="thin-scroll relative z-10 min-h-0 flex-1 overflow-y-auto">

          <Show when={!hasReal()}>
            <div class="mx-auto flex min-h-full w-full max-w-4xl flex-col justify-center px-6 py-8">
              <Greeting />
              <Show when={!voiceMode()}>
                <div class="mt-8">{composer()}</div>
              </Show>
              <Show when={notices().length > 0}>
                <div class="mt-4">
                  <Transcript msgs={notices()} onSpeak={(t) => void speak(t)} />
                </div>
              </Show>
            </div>
          </Show>

          <Show when={hasReal()}>

          <Show when={!bridgeOk()}>
            <p class="mx-auto mt-3 max-w-4xl px-6 text-center text-xs text-red-300/80">
              Bridge is down at <span class="font-mono">{bridgeUrl()}</span> — start{" "}
              <span class="font-mono">bridge.py</span> first. This window never spawns the model itself.
            </p>
          </Show>

          <Show when={busy()}>
            <p class="mt-3 animate-pulse text-center text-xs text-white/50">{phase()}…</p>
          </Show>

          <Show when={confirmAction()}>
            <div class="mx-auto mt-4 max-w-4xl rounded-[20px] bg-black/25 p-1.5 backdrop-blur-lg">
            <div class="rounded-2xl bg-[#241304]/80 p-4 text-white ring-1 ring-amber-300/30 backdrop-blur-xl">
              <p class="text-sm font-semibold text-amber-200">Confirm this action?</p>
              <pre class="thin-scroll mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-white/75">
                {JSON.stringify(confirmAction(), null, 2)}
              </pre>
              <div class="mt-3 flex gap-2">
                <button
                  onClick={() => {
                    sock.confirm(true);
                    setConfirmAction(null);
                  }}
                  class="rounded-xl bg-emerald-400 px-4 py-2 text-sm font-semibold text-black"
                >
                  Allow
                </button>
                <button
                  onClick={() => {
                    sock.confirm(false);
                    setConfirmAction(null);
                  }}
                  class="rounded-xl bg-white/10 px-4 py-2 text-sm text-white"
                >
                  Deny
                </button>
              </div>
            </div>
            </div>
          </Show>

            <Show when={showHistory()}>
              <div class="mx-auto mt-4 w-full px-6 pb-2 md:max-w-4xl md:px-0">
                <Transcript msgs={msgs()} onSpeak={(t) => void speak(t)} />
              </div>
            </Show>
          </Show>
        </div>

        <Show when={hasReal() && !voiceMode()}>
          <div class="relative z-10 shrink-0 px-6 pb-3 md:mx-auto md:w-full md:max-w-4xl md:px-0">{composer()}</div>
        </Show>

        <Show when={voiceMode()}>
          <div
            class="pointer-events-none absolute inset-x-0 bottom-0 z-30 backdrop-blur-lg"
            style={{
              "mask-image": "linear-gradient(to top, black 55%, transparent 100%)",
              "-webkit-mask-image": "linear-gradient(to top, black 55%, transparent 100%)",
            }}
          >
            <ReactVoiceMode
              processing={() => busy() || phase() !== "idle"}
              onMicError={(m) => push("err", `mic: ${m}`)}
            />
          </div>
        </Show>

        </section>
      </div>
      <div class="flex shrink-0 flex-col">
        <div
          class={`order-2 overflow-hidden transition-all duration-700 ease-in-out ${
            metricsOpen() ? "max-h-72 opacity-100" : "max-h-0 opacity-0"
          }`}
        >
          <div class="min-h-0 overflow-hidden">
            <div class="thin-scroll max-h-64 overflow-y-auto px-2 pb-2">            <div class="grid gap-3 md:grid-cols-3">
              <div class="dpanel rounded-2xl bg-black/45 p-4 ring-1 ring-white/10 shadow-2xl backdrop-blur">
                <p class="mb-3 text-[11px] uppercase tracking-widest text-white/40">Graphs</p>
                <div class="grid grid-cols-2 gap-3">
                  <Spark data={hist().map((p) => p.vram)} stroke="#38bdf8" label="VRAM %" />
                  <Spark data={hist().map((p) => p.gpu)} stroke="#34d399" label="GPU %" />
                  <Spark data={hist().map((p) => p.cpu)} stroke="#fbbf24" label="CPU %" />
                  <Spark data={hist().map((p) => p.ctx)} stroke="#a78bfa" label="CTX %" />
                </div>
              </div>
              <div class="dpanel rounded-2xl bg-black/45 p-4 text-white ring-1 ring-white/10 shadow-2xl backdrop-blur">
                <p class="mb-3 text-[11px] uppercase tracking-widest text-white/40">Info</p>
                <Show
                  when={metrics()}
                  fallback={<p class="text-center text-xs text-white/40">no metrics yet — is bridge.py up?</p>}
                >
                  {(m) => (
                    <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                      <p>
                        <span class="lbl">VRAM </span>
                        <span class="val font-mono">{Math.round(m().vram.allocated_mb)} / {Math.round(m().vram.total_mb)} MB</span>
                      </p>
                      <p>
                        <span class="lbl">RAM </span>
                        <span class="val font-mono">{Math.round(m().ram.used_mb)} / {Math.round(m().ram.total_mb)} MB</span>
                      </p>
                      <p>
                        <span class="lbl">Uptime </span>
                        <span class="val font-mono">{fmtDur(m().uptime_s)}</span>
                      </p>
                      <p>
                        <span class="lbl">Turns </span>
                        <span class="val font-mono">{m().turns.done} · last {m().turns.last_s.toFixed(1)}s</span>
                      </p>
                      <div class="col-span-2">
                        <div class="flex items-center justify-between">
                          <span class="lbl">Context</span>
                          <span class="val font-mono">
                            {ctxPct()?.tokens ?? 0} / {ctxPct()?.ctx ?? 0} ({Math.round((ctxPct()?.pct ?? 0) * 100)}%)
                          </span>
                        </div>
                        <div class="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10">
                          <div class="h-full rounded-full bg-sky-400" style={{ width: `${Math.min(100, Math.round((ctxPct()?.pct ?? 0) * 100))}%` }} />
                        </div>
                      </div>
                      <Show when={legs()}>
                        <p class="val col-span-2 font-mono text-[11px] opacity-70">
                          {legs()!.legs.vad} → {legs()!.legs.stt} → {legs()!.legs.llm} → {legs()!.legs.tts} ·{" "}
                          {legs()!.model.label}
                          {metrics()!.queue.pending > 0 ? ` · queued ${metrics()!.queue.pending}` : ""}
                        </p>
                      </Show>
                    </div>
                  )}
                </Show>
              </div>
              <div class="dpanel rounded-2xl bg-black/45 p-4 text-white ring-1 ring-white/10 shadow-2xl backdrop-blur">
                <p class="mb-3 text-[11px] uppercase tracking-widest text-white/40">Settings</p>
                <div class="space-y-2.5 text-xs">
                  <div class="flex items-center gap-2">
                    <span class="lbl w-14 shrink-0">Bridge</span>
                    <input
                      value={bridgeUrl()}
                      onInput={(e) => setBridgeUrl(e.currentTarget.value)}
                      onChange={(e) => setAPI(e.currentTarget.value)}
                      class="dpanel-input min-w-0 flex-1 rounded-lg px-2.5 py-1.5 font-mono text-[11px] outline-none ring-1 ring-white/15"
                    />
                    <button onClick={() => void refresh()} class="dpanel-btn shrink-0 rounded-lg px-2.5 py-1.5">
                      Retry
                    </button>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="lbl w-14 shrink-0">CWD</span>
                    <input
                      value={cwd()}
                      onInput={(e) => setCwd(e.currentTarget.value)}
                      class="dpanel-input min-w-0 flex-1 rounded-lg px-2.5 py-1.5 font-mono text-[11px] outline-none ring-1 ring-white/15"
                    />
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="lbl w-14 shrink-0">Model</span>
                    <div class="flex min-w-0 flex-1 flex-wrap gap-1.5">
                      <For each={modelChoices()}>
                        {(c) => (
                          <button
                            onClick={() => void sendText(`/model ${c.name}`)}
                            class={`rounded-lg px-2.5 py-1.5 font-mono text-[11px] ring-1 ring-white/15 transition ${
                              modelCurrent() === c.name || modelCurrent() === c.label
                                ? "bg-white/20 text-white"
                                : "dpanel-btn"
                            }`}
                          >
                            {c.label}
                          </button>
                        )}
                      </For>
                      <Show when={modelChoices().length === 0}>
                        <span class="lbl font-mono text-[11px]">{modelCurrent()}</span>
                      </Show>
                    </div>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="lbl w-14 shrink-0">Session</span>
                    <span class="val min-w-0 flex-1 truncate font-mono text-[11px]">{sessionId()}</span>
                    <button onClick={() => void sendText("/new")} class="dpanel-btn shrink-0 rounded-lg px-2.5 py-1.5">
                      New
                    </button>
                  </div>
                </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="winbar order-1 flex h-10 shrink-0 items-center justify-between px-2 text-white/60">
        <div class="flex min-w-0 items-center gap-2">
          <button
            title={metricsOpen() ? "Collapse metrics drawer" : "Expand metrics drawer"}
            aria-label={metricsOpen() ? "Collapse metrics drawer" : "Expand metrics drawer"}
            onClick={() => {
              setMetricsOpen((v) => !v);
              void refresh();
            }}
            class="rawicon winbar-btn flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white/70 transition-colors duration-200 hover:bg-white/10 hover:text-white"
          >
            <span innerHTML={metricsOpen() ? collapseSvg : expandSvg} class="flex h-3.5 w-3.5 items-center justify-center" />
          </button>
          <button
            title={theme() === "light" ? "Switch to dark" : "Switch to light"}
            aria-label="Toggle dark / light theme"
            onClick={toggleTheme}
            class="winbar-btn flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white/70 transition-colors duration-200 hover:bg-white/10 hover:text-white"
          >
            <Show
              when={theme() === "light"}
              fallback={
                <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round">
                  <circle cx="8" cy="8" r="3.2" />
                  <path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M12.6 3.4l-1.3 1.3M8.7 11.3l-1.3 1.3" />
                </svg>
              }
            >
              <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7z" />
              </svg>
            </Show>
          </button>
          <button
            title={voiceMode() ? "Exit voice mode" : "Enter voice mode"}
            aria-label="Toggle voice mode"
            onClick={() => (voiceMode() ? setVoiceMode(false) : enterVoiceMode())}
            class={`relative h-6 w-6 shrink-0 overflow-hidden rounded-full transition active:scale-95 ${
              voiceMode()
                ? "opacity-100 ring-2 ring-blue-400"
                : "opacity-70 ring-1 ring-white/20 hover:opacity-100"
            }`}
          >
            <WarpOrb />
          </button>
          <button
            title="Sample conversation"
            aria-label="Open sample conversation"
            onClick={() => void switchRecent(SAMPLE_SID)}
            class={`winbar-btn flex h-6 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-[11px] transition-colors duration-200 hover:bg-white/10 hover:text-white ${
              sessionId() === SAMPLE_SID ? "bg-white/10 text-white" : "text-white/70"
            }`}
          >
            <svg viewBox="0 0 20 20" class="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 3h7l3 3v11H5zM12 3v3h3M8.5 12h4M8.5 14.5h4" />
            </svg>
            <span>Sample</span>
          </button>
          <div class="group relative flex shrink-0 items-center">
            <div class="pointer-events-none absolute bottom-full left-0 z-30 mb-2 hidden max-w-64 whitespace-normal break-all rounded-xl bg-black/90 px-3 py-2 font-mono text-[11px] text-white/80 ring-1 ring-white/15 group-hover:block">
              {sessionId()}
            </div>
            <button
              title="Current conversation"
              onClick={() => setConvOpen((v) => !v)}
              class="winbar-btn flex h-6 max-w-44 items-center gap-1.5 rounded-full py-0 pl-2.5 pr-1.5 text-[11px] text-white/70 transition-colors duration-200 hover:bg-white/10 hover:text-white"
            >
              <span class="truncate">{sessionName()}</span>
              <svg
                viewBox="0 0 16 16"
                class={`h-3.5 w-3.5 shrink-0 transition-transform ${convOpen() ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
              >
                <path d="M4 6l4 4 4-4" />
              </svg>
            </button>
            <Show when={convOpen()}>
              <div class="absolute bottom-full left-0 z-30 mb-2 w-64 overflow-hidden rounded-2xl bg-black/90 ring-1 ring-white/15 backdrop-blur">
                <div class="flex items-center justify-between px-3 pb-1 pt-2.5">
                  <p class="text-[10px] uppercase tracking-widest text-white/40">Recent conversations</p>
                  <button onClick={() => void sendText("/new")} class="rounded-lg bg-white/10 px-2 py-0.5 text-[11px] text-white/70 hover:bg-white/20">
                    + New
                  </button>
                </div>
                <div class="thin-scroll max-h-52 overflow-y-auto p-1.5">
                  <Show when={recents().length === 0} fallback={
                    <For each={recents()}>
                      {(r) => (
                        <button
                          onClick={() => void switchRecent(r.id)}
                          class={`flex w-full flex-col gap-0.5 rounded-xl px-2.5 py-2 text-left transition hover:bg-white/10 ${
                            r.id === sessionId() ? "bg-white/10" : ""
                          }`}
                        >
                          <span class="truncate text-xs text-white/85">{r.name}</span>
                          <span class="truncate font-mono text-[10px] text-white/35">
                            {r.id} · {new Date(r.ts).toLocaleDateString()} {new Date(r.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </button>
                      )}
                    </For>
                  }>
                    <p class="px-2.5 py-2 text-xs text-white/35">no conversations yet</p>
                  </Show>
                </div>
              </div>
            </Show>
          </div>
        </div>
        <div class="flex min-w-0 shrink-0 items-center gap-2">
          <span class="barmini truncate font-mono text-[11px] text-white/30">{metricsMini()}</span>
          <Show when={isTauri()}>
            <div class="flex shrink-0 items-center gap-1">
            <button
              title="Minimize"
              aria-label="Minimize"
              onClick={() => void winAction("min")}
              class="winbar-btn flex h-7 w-10 items-center justify-center rounded-md text-white/55 transition-colors duration-200 hover:bg-white/10 hover:text-white"
            >
              <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                <path d="M3.5 8h9" />
              </svg>
            </button>
            <button
              title="Maximize / restore"
              aria-label="Maximize or restore"
              onClick={() => void winAction("max")}
              class="winbar-btn flex h-7 w-10 items-center justify-center rounded-md text-white/55 transition-colors duration-200 hover:bg-white/10 hover:text-white"
            >
              <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8">
                <rect x="3.5" y="3.5" width="9" height="9" rx="1.5" />
              </svg>
            </button>
            <button
              title="Close"
              aria-label="Close"
              onClick={() => void winAction("close")}
              class="winbar-btn winclose flex h-7 w-10 items-center justify-center rounded-md text-white/55 transition-colors duration-200 hover:bg-red-600 hover:text-white"
            >
              <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                <path d="M4 4l8 8M12 4l-8 8" />
              </svg>
            </button>
            </div>
          </Show>
        </div>
      </div>
      </div>
    </div>
  );
}
