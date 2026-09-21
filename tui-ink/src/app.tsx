import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import TextInput from "ink-text-input";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { API, TermSocket, ensureBackend, health, say, stopBackend, stt, type TurnEvent } from "./bridge.js";
import { NB, b64ToF32, envelope, pcmToF32, rmsLevel, spectrum } from "./audio.js";

/** Word-Jaccard similarity 0..1 (echo detection). */
export function similarity(a: string, b: string): number {
  const wa = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const wb = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  if (!wa.size || !wb.size) return 0;
  let inter = 0;
  for (const w of wa) if (wb.has(w)) inter++;
  return inter / Math.max(wa.size, wb.size);
}

/** Head of a reply for voicing; full text stays on screen/in memory. */
export function speakHead(full: string, limit = 280): string {
  const t = full.trim().replace(/\s+/g, " ");
  if (t.length <= limit) return t;
  const cut = t.slice(0, limit);
  const dot = cut.lastIndexOf(". ");
  const head = (dot > 120 ? cut.slice(0, dot + 1) : cut).trim();
  return `${head} … full output is on screen.`;
}

/** True when the transcript is basically a (partial) re-hearing of speech. */
export function isEcho(spoken: string[], heard: string): boolean {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
  const h = norm(heard);
  if (h.length < 4) return false;
  const hWords = h.split(" ").length;
  for (const s of spoken) {
    const t = norm(s);
    if (!t) continue;
    if (similarity(t, h) >= 0.6) return true;
    // whole heard phrase (2+ words) appears inside what we said -> echo
    if (hWords >= 2 && t.includes(h)) return true;
    // long heard phrase contains the head of what we said -> partial echo
    if (h.length >= 12 && t.includes(h.slice(0, Math.min(60, h.length)))) return true;
  }
  return false;
}

type Msg = { id: string; who: "you" | "agent" | "sys" | "act" | "obs" | "err" | "ask"; text: string };
type VizMode = "idle" | "user" | "agent";
type Mode = "auto" | "voice";
const MODES: Mode[] = ["auto", "voice"];

const uid = () => randomBytes(4).toString("hex");
const MODEL = "Qwen3-0.6B";




export function App({ seconds = 5 }: { seconds?: number }) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [sid, setSid] = useState(() => randomBytes(8).toString("hex"));
  const [cwd, setCwd] = useState(() => process.cwd());
  const [serverOk, setServerOk] = useState<boolean | null>(null);
  const [viz, setViz] = useState<{ mode: VizMode; bins: number[] }>({ mode: "idle", bins: new Array(NB).fill(0) });
  const [tick, setTick] = useState(0);
  const [pending, setPending] = useState<Record<string, unknown> | null>(null);
  const [mode, setMode] = useState<Mode>(() => {
    const m = process.env.VOICE_MODE;
    return m === "voice" ? "voice" : "auto";
  });
  const sock = useRef<TermSocket | null>(null);
  const busyRef = useRef(false);
  busyRef.current = busy;
  const modeRef = useRef<Mode>("auto");
  modeRef.current = mode;
  // Half-duplex: never listen while the agent is speaking, or the mic
  // re-ingests our own TTS and the conversation loops on itself.
  // Refcount (not boolean): overlapping playbacks must not clear each other.
  // lastSpeakEnd adds a drain margin for the ALSA/PipeWire tail that keeps
  // sounding after our timer ends.
  const speakCount = useRef(0);
  const speakingRef = useRef(false);
  const lastSpeakEnd = useRef(0);
  const lastSpokenRef = useRef<string[]>([]);
  const curPlayer = useRef<import("node:child_process").ChildProcess | null>(null);
  const incSpeak = () => {
    speakCount.current += 1;
    speakingRef.current = true;
  };
  const decSpeak = () => {
    speakCount.current = Math.max(0, speakCount.current - 1);
    if (speakCount.current === 0) {
      speakingRef.current = false;
      lastSpeakEnd.current = Date.now();
    }
  };
  const rememberSpoken = (t: string) => {
    if (!t.trim()) return;
    lastSpokenRef.current = [...lastSpokenRef.current.slice(-1), t];
  };

  async function waitSilent(timeoutMs = 30000): Promise<void> {
    const t0 = Date.now();
    while (
      (speakingRef.current || Date.now() - lastSpeakEnd.current < 1200) &&
      Date.now() - t0 < timeoutMs
    ) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }

  const push = useCallback((who: Msg["who"], text: string) => {
    setMsgs((m) => [...m.slice(-199), { id: uid(), who, text }]);
  }, []);

  const calm = useCallback(() => setViz({ mode: "idle", bins: new Array(NB).fill(0) }), []);

  // idle wave
  useEffect(() => {
    const t = setInterval(() => setTick((x) => x + 1), 120);
    return () => clearInterval(t);
  }, []);
  void tick;

  // bridge health dot
  useEffect(() => {
    let stop = false;
    const check = async () => {
      try {
        const h = await health();
        if (!stop) setServerOk(h.ok);
      } catch {
        if (!stop) setServerOk(false);
      }
    };
    void check();
    const t = setInterval(check, 10000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  const playAudio = useCallback(
    async (wav: Float32Array, sr: number) => {
      // A new playback kills any stray previous player: two overlapping
      // voices sound like reverb and double the echo surface.
      try {
        curPlayer.current?.kill("SIGKILL");
      } catch {
        /* ignore */
      }
      curPlayer.current = null;
      incSpeak();
      let mine: import("node:child_process").ChildProcess | null = null;
      try {
        const frames = envelope(wav);
        const dur = wav.length / sr;
        const child = spawn("aplay", ["-q", "-f", "S16_LE", `-r${sr}`, "-c1", "-t", "raw", "-"], {
          stdio: ["pipe", "ignore", "ignore"],
        });
        mine = child;
        curPlayer.current = child;
        // Swallow async stdin errors: if aplay dies (or was just killed as
        // a stray previous player), the write below fails with EPIPE on the
        // socket — without a listener that crashes the whole app.
        child.stdin?.on("error", () => {});
        const procDone = new Promise<void>((resolve) => {
          const to = setTimeout(() => resolve(), dur * 1000 + 5000);
          child.on("error", () => {
            clearTimeout(to);
            resolve();
          });
          child.on("close", () => {
            clearTimeout(to);
            resolve();
          });
        });
        try {
          const pcm = Buffer.from(wav.buffer, wav.byteOffset, wav.length * 4);
          // f32 -> s16 for raw stdin
          const s16 = Buffer.alloc(wav.length * 2);
          for (let i = 0; i < wav.length; i++) {
            const v = Math.max(-1, Math.min(1, wav[i] ?? 0));
            s16.writeInt16LE(Math.round(v * 32767), i * 2);
          }
          child.stdin?.write(s16);
          child.stdin?.end();
        } catch {
          /* ignore */
        }
        setViz({ mode: "agent", bins: frames[0] ?? [] });
        const per = Math.max(50, (dur / frames.length) * 1000);
        // Wait for BOTH the animation clock and the real player, so the
        // mic never opens over trailing audio.
        await Promise.all([
          (async () => {
            for (const f of frames) {
              setViz({ mode: "agent", bins: f });
              await new Promise((r) => setTimeout(r, per));
            }
          })(),
          procDone,
        ]);
      } finally {
        if (mine && curPlayer.current === mine) curPlayer.current = null;
        decSpeak();
        calm();
      }
    },
    [calm]
  );

  // Spoken replies stay SHORT (see speakHead): full outputs live in the
  // chat bubble; voicing multi-KB listings is unlistenable echo surface.
  function speakText(full: string): string {
    return speakHead(full);
  }

  const speakAgent = useCallback(
    async (text: string) => {
      // Voice pipeline (TTS) runs in voice mode only; auto mode is text.
      if (modeRef.current !== "voice") return;
      const said = speakText(text);
      rememberSpoken(said);
      // Mark speaking for the WHOLE operation including TTS synthesis:
      // otherwise a v-press during the HTTP call opens the mic and the
      // reply starts mid-recording.
      incSpeak();
      try {
        const r = await say(said);
        if (r.kind !== "audio" || !r.wav_b64) throw new Error(r.message ?? "tts unavailable");
        await playAudio(b64ToF32(r.wav_b64), r.sr ?? 24000);
      } catch (e) {
        push("err", `server TTS failed (${String(e)}) — reply shown as text only`);
      } finally {
        decSpeak();
      }
    },
    [playAudio, push]
  );

  const chainRef = useRef<() => void>(() => {});
  const voiceConfirmRef = useRef<(a: Record<string, unknown>) => void>(() => {});

  // Live loop is VOICE-mode only: auto mode is plain text+LLM with no
  // voice involved (no mic, no TTS). Voice chains listen→turn→speak→listen.
  const chainNext = useCallback(() => {
    if (busyRef.current || modeRef.current !== "voice") return;
    void (async () => {
      await waitSilent();
      if (!busyRef.current && modeRef.current === "voice") chainRef.current();
    })();
  }, []);

  // persistent socket
  useEffect(() => {
    const s = new TermSocket();
    sock.current = s;
    s.onEvent = (e: TurnEvent) => {
      if (e.event === "action") {
        const a = (e as { action: Record<string, unknown> }).action ?? {};
        push("act", `▸ ${String(a["action"] ?? "?")}: ${String(a["command"] ?? a["path"] ?? "")}`);
      } else if (e.event === "observation") {
        push("obs", String((e as { observation: string }).observation ?? "").slice(0, 800));
      } else if (e.event === "chat") {
        const reply = String((e as { reply: string }).reply ?? "");
        push("agent", reply);
        void speakAgent(reply);
      } else if (e.event === "audio") {
        // Bridge TTS audio is voiced in voice mode only (auto = text).
        if (modeRef.current === "voice") {
          const ev = e as { wav_b64: string; sr: number };
          void playAudio(b64ToF32(ev.wav_b64), ev.sr ?? 24000);
        }
      } else if (e.event === "summary") {
        setBusy(false);
        chainNext();
      } else if (e.event === "error") {
        push("err", `error: ${String((e as { message: string }).message ?? "")}`);
        setBusy(false);
      }
    };
    s.onConfirm = (action) => {
      if (modeRef.current === "voice") voiceConfirmRef.current(action);
      else setPending(action);
    };
    // Single local app: reuse a healthy backend if one is already up,
    // else spawn bridge.py as our own child (killed with us on exit).
    // server.py is never involved.
    const boot = async () => {
      const want = await ensureBackend((m) => push("sys", m));
      if (!want) {
        push("err", "local agent backend failed to start (see terminal above)");
        return;
      }
      try {
        await s.connect();
      } catch {
        push("err", "bridge unreachable after start — retrying on next turn");
      }
    };
    void boot();
    return () => {
      try {
        s && sock.current === s && (sock.current = null);
      } catch {
        /* ignore */
      }
      stopBackend();
    };
  }, [push, speakAgent, playAudio, chainNext]);

  const runText = useCallback(
    (text: string) => {
      if (busyRef.current || !sock.current?.connected) {
        if (!sock.current?.connected) push("err", "not connected to bridge yet");
        return;
      }
      setBusy(true);
      push("you", text);
      push("sys", "⬡ Cooking…");
      sock.current.turn(text, sid, cwd);
    },
    [push, sid, cwd]
  );

  // Mic primitive: records `secs` of PCM, drives the viz, returns null
  // when nothing usable was captured. No STT, no turn — composable.
  const listenOnce = useCallback(
    async (secs: number): Promise<Buffer | null> => {
      const chunks: Buffer[] = [];
      let child;
      try {
        child = spawn("arecord", ["-q", "-f", "S16_LE", "-r16000", "-c1", "-t", "raw", "-"], {
          stdio: ["ignore", "pipe", "ignore"],
        });
      } catch {
        push("err", "arecord missing — type your command instead");
        return null;
      }
      setViz({ mode: "user", bins: new Array(NB).fill(0) });
      child.stdout?.on("data", (d: Buffer) => {
        chunks.push(d);
        setViz({ mode: "user", bins: spectrum(pcmToF32(d)) });
        void rmsLevel(d);
      });
      await new Promise((r) => setTimeout(r, secs * 1000));
      child.kill("SIGTERM");
      calm();
      const pcm = Buffer.concat(chunks);
      return pcm.length >= 3200 ? pcm : null;
    },
    [push, calm]
  );

  // One hands-free voice step: wait for our own playback to end, listen,
  // STT, discard self-echo, run turn (chains again at summary in loops).
  // `force` bypasses the busy gate because a summary means the previous
  // turn already ended (ref updates lag state).
  const quietStreak = useRef(0);
  const voiceStep = useCallback(
    async (secs: number, force = false) => {
      if (modeRef.current !== "voice") return;
      if (!force && busyRef.current) return;
      await waitSilent();
      if (!force && busyRef.current) return;
      setBusy(true);
      // Chained (force) steps keep the live loop going through silence,
      // capped so a dead room doesn't spin forever.
      const keepAlive = () => {
        setBusy(false);
        if (force) {
          quietStreak.current += 1;
          if (quietStreak.current > 12) {
            quietStreak.current = 0;
            push("sys", "quiet for a while — Tab to resume the live loop.");
            return;
          }
          chainNext();
        }
      };
      push("sys", "recording — speak now");
      const pcm = await listenOnce(secs);
      if (!pcm) {
        push("sys", "nothing recorded.");
        keepAlive();
        return;
      }
      try {
        const r = await stt(pcm);
        if (r.kind !== "text" || !r.text?.trim()) {
          push("sys", `STT empty (${r.message ?? "silence?"})`);
          keepAlive();
          return;
        }
        // Echo guard: the mic may still catch our speaker (no headphones).
        // Matches full or partial re-hearings of our last replies.
        if (isEcho(lastSpokenRef.current, r.text)) {
          push("sys", "heard my own voice — discarded.");
          keepAlive();
          return;
        }
        quietStreak.current = 0;
        setBusy(false);
        runText(r.text);
      } catch (e) {
        push("err", `STT failed: ${String(e)}`);
        setBusy(false);
      }
    },
    [push, calm, listenOnce, runText]
  );

  // Voice-mode confirm gate: speak the question, listen for yes/no.
  const voiceConfirm = useCallback(
    async (action: Record<string, unknown>) => {
      const what = String(action["command"] ?? action["path"] ?? action["action"] ?? "?");
      push("ask", `Should I run: ${what}? Say yes or no.`);
      await speakAgent(`Should I run ${what.slice(0, 120)}? Say yes or no.`);
      const pcm = await listenOnce(4);
      let ok = false;
      if (pcm) {
        try {
          const r = await stt(pcm);
          const t = (r.text ?? "").toLowerCase();
          ok = /\b(yes|yeah|yep|confirm|ok|okay|sure|do it|go ahead)\b/.test(t);
          push("sys", `heard: "${r.text ?? ""}" → ${ok ? "confirmed ✓" : "denied."}`);
        } catch {
          push("sys", "confirm listen failed → denied.");
        }
      } else {
        push("sys", "no answer → denied.");
      }
      sock.current?.confirm(ok);
    },
    [push, speakAgent, listenOnce]
  );

  chainRef.current = () => {
    void voiceStep(seconds, true);
  };
  voiceConfirmRef.current = (a) => {
    void voiceConfirm(a);
  };

  const onSubmit = useCallback(
    (text: string) => {
      setValue("");
      const t = text.trim();
      if (!t || busyRef.current) return;
      if (t === "/quit" || t === "/q") return exit();
      if (t === "/new") {
        setSid(randomBytes(8).toString("hex"));
        push("sys", "new session");
        return;
      }
      if (t === "/clear") {
        setMsgs([]);
        return;
      }
      if (t.startsWith("/cwd")) {
        const p = t.split(/\s+/, 2)[1];
        if (p) {
          setCwd(p);
          push("sys", `cwd ${p}`);
        }
        return;
      }
      if (t.startsWith("/voice")) {
        if (modeRef.current !== "voice") {
          push("sys", "voice lives in voice mode — Tab to switch.");
          return;
        }
        const s = Number(t.split(/\s+/, 2)[1]) || seconds;
        void voiceStep(s);
        return;
      }
      if (t === "/mode") {
        push("sys", `mode: ${modeRef.current} (Tab toggles auto ↔ voice)`);
        return;
      }
      if (t.startsWith("/mode ")) {
        const m = t.split(/\s+/, 2)[1] as Mode;
        if (m === "auto" || m === "voice") {
          setMode(m);
          push("sys", `mode → ${m}`);
        } else push("err", "mode must be auto|voice");
        return;
      }
      if (t === "/help") {
        push("sys", "/new /cwd PATH /clear /voice [sec] /mode [m] /help /quit · Tab mode · v voice · y/n confirm");
        return;
      }
      if (t.startsWith("/")) {
        push("err", `unknown ${t} — /help`);
        return;
      }
      runText(t);
    },
    [exit, push, voiceStep, runText, seconds]
  );

  useInput((input, key) => {
    if (key.tab) {
      // Tab cycles modes everywhere (also while typing — Tab never edits).
      setMode((m) => {
        const n = MODES[(MODES.indexOf(m) + 1) % MODES.length] ?? "auto";
        push("sys", `mode → ${n}${n === "voice" ? " (live voice pipeline)" : " (text + LLM, no voice)"}`);
        return n;
      });
      return;
    }
    if (pending) {
      if (input === "y" || input === "Y") {
        push("sys", "confirmed ✓");
        setPending(null);
        sock.current?.confirm(true);
      } else if (input === "n" || input === "N" || key.escape) {
        push("sys", "denied.");
        setPending(null);
        sock.current?.confirm(false);
      }
      return;
    }
    if ((input === "v" || input === "V") && value.trim() === "" && !busyRef.current) {
      if (modeRef.current !== "voice") {
        push("sys", "voice lives in voice mode — Tab to switch.");
        return;
      }
      void voiceStep(seconds);
    }
  });

  const rows = process.stdout.rows ?? 30;
  const convH = Math.max(6, rows - 9);

  // Agent voice as three vertical bars: middle bigger, sides smaller,
  // same thickness (3 cells), bright white. Driven by low/mid/high bands
  // of the agent levels; gentle idle pulse otherwise. No user visualizer.
  const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + (b ?? 0), 0) / xs.length : 0);
  const bins = viz.bins;
  const live = viz.mode === "agent";
  const t = tick / 8;
  // Floors keep all three bars visible even in short terminals / quiet audio.
  const bands = live
    ? [
        Math.max(avg(bins.slice(0, 8)), 0.14),
        Math.min(1, Math.max(avg(bins.slice(8, 16)) * 1.35, 0.2)),
        Math.max(avg(bins.slice(16, 24)), 0.14),
      ]
    : [
        0.26 + 0.1 * Math.abs(Math.sin(t)),
        0.4 + 0.14 * Math.abs(Math.sin(t + 1.3)),
        0.26 + 0.1 * Math.abs(Math.sin(t + 2.6)),
      ];
  // VoiceRT mark: three bars, middle tallest, sides ~60%, same thickness.
  // Bottoms locked on one baseline, only the tops move. Big and tall.
  const BH = 12;
  const BW = 3;
  const GAP = "  ";
  const barLine = (r: number) =>
    bands.map((v) => ((v ?? 0) * BH >= r ? "█".repeat(BW) : " ".repeat(BW)).valueOf()).join(GAP);
  const trioWidth = BW * 3 + GAP.length * 2;
  const cols0 = process.stdout.columns ?? 100;
  const rw0 = Math.max(10, Math.floor(cols0 * 0.18) - 2);
  const off = Math.max(0, Math.floor((rw0 - trioWidth) / 2));

  const colorFor = (w: Msg["who"]) =>
    w === "you" ? "cyan" : w === "agent" ? "white" : w === "err" ? "red" : w === "ask" ? "yellow" : "gray";
  const labelFor = (w: Msg["who"]) =>
    w === "you" ? "› " : w === "agent" ? "" : w === "ask" ? "⬡ Confirm? " : "";

  // Right panel: visualizer ONLY — no labels, no status. Borderless,
  // filled with the same faint tone as the left border. Ink 5 has no Box
  // background, so every row is a full-width Text carrying the fill.
  const cols = process.stdout.columns ?? 100;
  const rw = Math.max(10, Math.floor(cols * 0.18) - 2);
  const pad = (s: string) => (s + " ".repeat(rw)).slice(0, rw);
  type RRow = { text: string; color?: string; bold?: boolean };
  const rightRows: RRow[] = [{ text: "" }];
  for (let r = BH; r >= 1; r--) {
    rightRows.push({ text: " ".repeat(off) + barLine(r), color: "white", bold: true });
  }
  while (rightRows.length < rows - 1) rightRows.push({ text: "" });

  return (
    <Box flexDirection="row" height={rows - 1}>
      <Box flexDirection="column" width="82%" height={rows - 1} borderStyle="single" borderColor="#232327" paddingX={1}>
        <Text bold color="#5a5a60">
          {" conversation "}
        </Text>
        <Box flexGrow={1} flexDirection="column" overflow="hidden">
          {msgs.slice(-convH + 1).map((m) => (
            <Text key={m.id} color={colorFor(m.who)} wrap="wrap">
              {labelFor(m.who)}
              {m.text}
            </Text>
          ))}
        </Box>
        <Box borderStyle="single" borderColor="white">
          <Text color="white">
            {" "}
            {mode === "voice" ? <Text color="#3b82f6">voice</Text> : mode}
            {" -> "}
          </Text>
          <TextInput
            value={value}
            onChange={(v) => setValue(v.replace(/\t/g, ""))}
            onSubmit={onSubmit}
            placeholder={
              mode === "voice" ? "live — just speak, no typing needed" : "Add a follow-up  ( / for commands · v for voice )"
            }
          />
        </Box>
        <Text color="white">
          {MODEL} · {mode} · session {sid.slice(0, 8)} · {cwd} · {serverOk === null ? "…" : serverOk ? "●" : "○ bridge down"} · Tab mode · v voice
        </Text>
        {pending ? (
          <Text color="yellow" wrap="wrap">
            Confirm {String(pending["action"] ?? "?")}: {String(pending["command"] ?? pending["path"] ?? "")} (y/n)
          </Text>
        ) : null}
      </Box>
      <Box flexDirection="column" width="18%">
        {rightRows.map((l, i) => (
          <Text key={i} backgroundColor="#232327" color={l.color ?? "white"} bold={l.bold}>
            {pad(l.text)}
          </Text>
        ))}
      </Box>
    </Box>
  );
}
