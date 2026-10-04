import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { TextAttributes, type MouseEvent } from "@opentui/core";
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/solid";
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { DeepSocket, TermSocket, ensureBackend, fetchTitle, getLegs, getModel, health, say, sayStream, stopBackend, stt, sttInterim, switchModel, type TurnEvent } from "./bridge.js";
import { printEpilogue, RT_COL, WORDMARK } from "./epilogue.js";
import { markdownStyle } from "./markdown.js";
import { Panel, type Todos } from "./panel.js";
import { newSessionId, sessionName } from "./session.js";
import { phaseLabel, spinnerFrame } from "./spinner.js";
import { SUGGESTIONS } from "./suggestions.js";
import {
  SHADES,
  initialShade,
  isShade,
  shadeHex,
  withShade,
  type Shade,
  type Theme,
} from "./theme.js";
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

type Msg = {
  id: string;
  who: "you" | "agent" | "sys" | "act" | "obs" | "err" | "ask" | "think" | "deny" | "compact";
  text: string;
  // Wall clock (bridge `ts`): thoughts and tool calls carry it.
  ts?: number;
  // Queue position while the session lock is held (queued event).
  queued?: number;
  // Tool call completed (observation arrived) — the ✓ sign.
  done?: boolean;
};
type VizMode = "idle" | "user" | "agent";
type Mode = "auto" | "voice";
const MODES: Mode[] = ["auto", "voice"];

// The command palette. `name` is what onSubmit dispatches on; `hint` is the
// one-line summary shown in the menu. `args` is rendered into the hint so a
// command that takes an argument advertises its own usage (see /cwd PATH).
// This list is the menu's single source of truth — /help reads from it too,
// so a new command can't be added here and forgotten there.
const COMMANDS: { name: string; hint: string }[] = [
  { name: "/new", hint: "start a fresh session" },
  { name: "/cwd", hint: "/cwd PATH — set the working directory" },
  { name: "/clear", hint: "clear the transcript" },
  { name: "/model", hint: "/model [name] — list or switch the model" },
  { name: "/mode", hint: "/mode [auto|voice] — show or set the mode" },
  { name: "/mic", hint: "/mic [on|off] — toggle the microphone" },
  { name: "/voice", hint: "/voice [sec] — record a voice turn" },
  { name: "/shade", hint: "/shade [name|none] — monotonic accent hue" },
  { name: "/settings", hint: "show legs, models and tool policy" },
  { name: "/dictate", hint: "/dictate [sec] — voice dictation input" },
  { name: "/opencode", hint: "/opencode [dir] — spawn opencode here" },
  { name: "/template", hint: "show 40-turn conversation template" },
  { name: "/panel", hint: "/panel — show or hide the dashboard" },
  { name: "/help", hint: "show this list" },
  { name: "/quit", hint: "/quit (or /q) — leave" },
];

/** Levenshtein distance, used only to suggest a command after a typo. */
function editDistance(a: string, b: string): number {
  const prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let last = prev[0] ?? 0;
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j] ?? 0;
      prev[j] = Math.min(
        (prev[j] ?? 0) + 1,
        (prev[j - 1] ?? 0) + 1,
        last + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      last = tmp;
    }
  }
  return prev[b.length] ?? 0;
}

const uid = () => randomBytes(4).toString("hex");
const MODEL_FALLBACK = "bonsai-2-27b-ptq1_0";

// Display caps: a single unbounded message (multi-KB listing, long
// thinking trace) wraps into dozens of terminal rows and used to blow
// past the fixed-height layout and break the whole app. Truncate at
// push time; the full text stays in session memory on the backend.
const CAP_FOR: Record<Msg["who"], number> = {
  you: 2000,
  agent: 2000,
  sys: 500,
  act: 300,
  obs: 800,
  err: 500,
  ask: 500,
  think: 800,
  deny: 500,
  compact: 500,
};
function capText(who: Msg["who"], text: string): string {
  const t = String(text ?? "");
  const cap = CAP_FOR[who] ?? 500;
  if (t.length <= cap) return t;
  return t.slice(0, cap) + ` …[${t.length - cap} chars more]`;
}

export function App(props: { seconds?: number; theme: Theme; themeNote?: string; initialSid?: string }) {
  const { initialSid } = props;
  const seconds = props.seconds ?? 5;
  // Monotonic shade: one hue for every chromatic role, switched live by
  // /shade. null = the original multi-hue palette. The backdrop never moves.
  const [shade, setShade] = createSignal<Shade | null>(initialShade());
  // Memo, not a plain const: the palette has to be recomputed when /shade
  // flips it, otherwise every `theme.` below would keep the boot-time colors.
  const theme = createMemo(() => withShade(props.theme, shade()));
  // Shared markdown style (one native handle for all assistant bubbles).
  const mdStyle = markdownStyle();
  // "RT" half of the wordmark: live shade color, or the plain accent when
  // unshaded. Same split as the farewell card (see epilogue.ts).
  const rtColor = () => {
    const s = shade();
    return s ? shadeHex(s, theme().mode) : theme().accent;
  };
  // Header voice level 0..1: live mic spectrum average while the user
  // side is audible, else 0 (bar collapses — it grows/shrinks with voice).
  const micLevel = () => {
    const v = viz();
    if (!micOn() || v.mode !== "user") return 0;
    const b = v.bins;
    if (!b.length) return 0;
    return Math.min(1, Math.max(0, b.reduce((a, x) => a + (x ?? 0), 0) / b.length));
  };
  const BAR_MAX = 20;
  const barW = () => Math.round(micLevel() * BAR_MAX);
  // VAD dot: only present while speech is detected (appears = listening).
  const vadLive = () => micOn() && speech();
  const renderer = useRenderer();
  const dims = useTerminalDimensions();
  const termRows = () => dims().height;
  const termCols = () => dims().width;

  const [msgs, setMsgs] = createSignal<Msg[]>([]);
  const [value, setValue] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [sid, setSid] = createSignal(initialSid ?? newSessionId());
  const [cwd, setCwd] = createSignal(process.cwd());
  const [serverOk, setServerOk] = createSignal<boolean | null>(null);
  const [modelLabel, setModelLabel] = createSignal<string>(MODEL_FALLBACK);
  const [viz, setViz] = createSignal<{ mode: VizMode; bins: number[] }>({ mode: "idle", bins: new Array(NB).fill(0) });
  const [tick, setTick] = createSignal(0);
  const [pending, setPending] = createSignal<Record<string, unknown> | null>(null);
  // Phase line above the input: what the agent is doing right now.
  const [phase, setPhase] = createSignal("");
  // Scrollback: PgUp/PgDn shifts the visible history window (0 = live).
  const [histOff, setHistOff] = createSignal(0);
  // Left dashboard visibility: hidden by default, /panel brings it back
  // (widget included — the whole column unmounts and returns together).
  const [showPanel, setShowPanel] = createSignal(false);
  // Effective left width in columns: 0 while hidden (full-width chat).
  const leftW = () => (showPanel() ? Math.floor(termCols() * 0.22) : 0);
  const [mode, setMode] = createSignal<Mode>(process.env.VOICE_MODE === "voice" ? "voice" : "auto");
  // Mic master switch: OFF pauses all voice capture (v, /voice, live loop).
  const [micOn, setMicOn] = createSignal(true);
  const [streamText, setStreamText] = createSignal("");
  // Dashboard state the left panel renders.
  const [todos, setTodos] = createSignal<Todos>([]);
  // VAD: speech in the current mic chunk (local energy gate).
  const [speech, setSpeech] = createSignal(false);
  // Dictation: a recording is in progress, plus its elapsed time.
  const [dictating, setDictating] = createSignal(false);
  const [recSeconds, setRecSeconds] = createSignal(0);
  // Interim STT draft: the live user bubble while speaking.
  const [draft, setDraft] = createSignal("");
  // Status line above the input: every former `sys` log lives here now, so
  // the transcript keeps only the conversation (user/agent/thoughts/tools).
  // Last message wins; capped to one row. Backend warm logs are the one
  // exception: they go to the left dashboard (backendNote), never here.
  const [status, setStatus] = createSignal("");
  const [backendNote, setBackendNote] = createSignal("");
  function note(text: string) {
    const t = String(text ?? "").trim();
    if (!t) return;
    setStatus(t.split("\n")[0]?.slice(0, 200) ?? "");
  }
  // Command palette: index of the highlighted row, and whether the user has
  // dismissed it with escape (dismissed stays closed until the text stops
  // looking like a command, so escaping doesn't pop it back on every keypress).
  const [cmdSel, setCmdSel] = createSignal(0);
  const [cmdDismissed, setCmdDismissed] = createSignal(false);

  // --- mutable (non-reactive) state: never rendered, only read in callbacks --
  let sock: DeepSocket | TermSocket | null = null;
  // Both sockets carry the confirm gate now (/deep got it too).
  function confirmSock(ok: boolean) {
    sock?.confirm(ok);
  }
  // Op of the last action — web/fetch observations collapse to a marker.
  let lastOp = "";
  let micRef = true;
  // Half-duplex: never listen while the agent is speaking, or the mic
  // re-ingests our own TTS and the conversation loops on itself.
  // Refcount (not boolean): overlapping playbacks must not clear each other.
  // lastSpeakEnd adds a drain margin for the ALSA/PipeWire tail.
  let speakCount = 0;
  let speaking = false;
  let lastSpeakEnd = 0;
  let lastSpoken: string[] = [];
  let curPlayer: ChildProcess | null = null;
  // True while listenOnce owns the mic (voice turn / spoken confirm).
  let recording = false;
  let micMon: ChildProcess | null = null;
  let streamBuf = "";
  let streamDirty = false;
  let turnChatCount = 0;
  let quietStreak = 0;
  let booted = false;
  // Dictation timers: elapsed-second clock + VAD speech decay.
  let recTimer: ReturnType<typeof setInterval> | null = null;
  let speechTimer: ReturnType<typeof setTimeout> | null = null;
  // First non-slash utterance this session (typed or spoken). Kept as the
  // offline fallback name and as the "was anything said at all" flag: an
  // empty value means no farewell card.
  let firstUtterance = "";
  // Model-generated two-word name (bridge `title` event, or a resumed
  // session's persisted one). Preferred over firstUtterance when present.
  const [sessionTitle, setSessionTitle] = createSignal("");

  /** VAD speech gate: true while chunks keep arriving above the
   *  room-hiss threshold, decays to false 300ms after the last. */
  function noteSpeech(now: boolean) {
    if (!now) return;
    setSpeech(true);
    if (speechTimer) clearTimeout(speechTimer);
    speechTimer = setTimeout(() => setSpeech(false), 300);
  }

  function push(who: Msg["who"], text: string, ts?: number) {
    const capped = capText(who, text);
    if (!capped.trim()) return;
    setMsgs((m) => {
      // Dedupe: the backend may re-emit an identical trace (retried
      // step, overlapping socket) — never render the same bubble twice.
      const last = m[m.length - 1];
      if (last && last.who === who && last.text === capped) return m;
      return [...m.slice(-199), { id: uid(), who, text: capped, ts }];
    });
  }
  // Template conversation for visualizing every chat bubble + scrollback:
  // 20 user turns with thinking traces (one timestamped), markdown agent
  // replies, pending/✓/✗ tool calls, observations, confirm gates, a deny,
  // an error, a compaction and a queued badge — rendered exactly like live
  // transcript lines. Long enough to overflow any terminal so PgUp/PgDn
  // scrolling is demonstrable. Re-seedable via /template.
  function seedTemplate() {
    note("Template preview — every bubble type (/clear to dismiss):");
    // Marks the last tool-call line done (✓) or denied (✗), like the live
    // observation/deny events do.
    const markLastAct = (done: boolean) =>
      setMsgs((m) => {
        for (let i = m.length - 1; i >= 0; i--) {
          if (m[i]!.who === "act") {
            if (m[i]!.done !== undefined) break;
            const copy = m[i]!;
            return [...m.slice(0, i), { ...copy, done }, ...m.slice(i + 1)];
          }
          if (m[i]!.who === "you") break;
        }
        return m;
      });
    // Tags the last user bubble with its queue position, like the live
    // `queued` event does while the session lock is held.
    const tagQueued = (pos: number) =>
      setMsgs((m) => {
        const i = m.map((x) => x.who).lastIndexOf("you");
        if (i < 0) return m;
        const copy = m[i]!;
        return [...m.slice(0, i), { ...copy, queued: pos }, ...m.slice(i + 1)];
      });
    const now = Math.floor(Date.now() / 1000);
    // Turn 1: user asks, agent surveys and plans (todos).
    push("you", "Can you help me organize the files in this directory?");
    push("think", "Listing files first, then grouping by type.", now);
    push("act", "▸ glob: *.* in /current");
    push("obs", "↳ glob evidence (42 chars — agent sees it, hidden here)");
    push("act", "todos: ✓ List files | ● Group by type | ○ Move files");
    push("agent", "Found 12 files. I'll group them: images, docs, code.");
    // Turn 2: user directs, agent executes shell tool calls.
    push("you", "Put images in img/, docs in docs/, code in src/");
    push("think", "Creating directories, then moving by extension.");
    push("act", "▸ shell: mkdir -p img docs src");
    push("act", "▸ shell: mv *.png *.jpg img/; mv *.md *.txt docs/");
    push("obs", "moved 9 files, 3 left");
    push("agent", "Done — 9 files moved. 3 misc files left over.");
    // Turn 3: user asks about leftovers, agent re-scans.
    push("you", "What are the leftover files?");
    push("act", "▸ glob: * in /current");
    push("obs", "↳ glob evidence (15 chars — agent sees it, hidden here)");
    push("agent", "notes.bak, todo.txt and data.csv — move them to docs/?");
    // Turn 4: user confirms, agent finishes and closes the todos.
    push("you", "Yes, move them to docs/ and finish up.");
    push("act", "▸ shell: mv notes.bak todo.txt data.csv docs/");
    push("act", "todos: ✓ List files | ✓ Group by type | ✓ Move files");
    push("agent", "All organized. img/ has 4, docs/ has 6, src/ has 5.");
    // Turn 5: user wants to see the result, agent lists the tree.
    push("you", "Show me the final tree.");
    push("think", "Listing directories one level deep.");
    push("act", "▸ shell: find . -maxdepth 2 | sort");
    push("obs", "img/ docs/ src/, 15 files total");
    push("agent", "img/ 4, docs/ 6, src/ 5. Flat and tidy.");
    // Turn 6: user wants docs, agent writes a README (markdown showcase:
    // headings, bold, code fence and list render through the markdown bubble).
    push("you", "Write a short README for this layout.");
    push("think", "Describing each directory in one line.");
    push("act", "▸ shell: writing README.md");
    push("obs", "wrote 6 lines to README.md");
    push("agent", "**README.md** written:\n- `img/` — photos\n- `docs/` — notes\n- `src/` — code\n```sh\nls img docs src\n```");
    // Turn 7: user worries about secrets, agent greps.
    push("you", "Did I leave any secrets or keys behind?");
    push("think", "Grepping for key, token and secret patterns.");
    push("act", "▸ shell: grep -ri 'key\\|token\\|secret' .");
    push("obs", "no matches");
    push("agent", "Clean — no secrets or keys found.");
    // Turn 8: user wants backups gone (confirm-gated action + queued badge
    // + a denied tool call, so ✓/▸/✗ and `[queued #N]` all render).
    push("you", "Remove the backup files.");
    tagQueued(2);
    push("ask", "rm notes.bak");
    push("act", "▸ shell: rm notes.bak");
    push("obs", "removed 1 file");
    markLastAct(true);
    push("act", "▸ shell: sudo reboot");
    markLastAct(false);
    push("deny", "✗ denied: policy (sudo)");
    push("agent", "Backups removed. One less clutter.");
    // Turn 9: user asks about size, agent measures (error bubble included).
    push("you", "How big is everything now?");
    push("act", "▸ shell: du -sh img docs src");
    push("obs", "img 12M, docs 3M, src 8M");
    markLastAct(true);
    push("err", "error: TTS voice unavailable — reply shown as text only");
    push("agent", "23M total — images are the bulk of it.");
    // Turn 10: user wraps up arc one, agent closes out.
    push("you", "Thanks, that's tidy. One more round?");
    push("act", "todos: ✓ List files | ✓ Group by type | ✓ Move files");
    push("agent", "Sure — repo hygiene next? Say the word.");
    // Turn 11: git init.
    push("you", "Init a git repo here.");
    push("think", "One command, then confirming the branch.");
    push("act", "▸ shell: git init -b main");
    push("obs", "initialized empty repo on main");
    push("agent", "Repo initialized on branch main.");
    // Turn 12: commit the layout.
    push("you", "Commit this layout.");
    push("think", "Staging everything, then committing.");
    push("act", "▸ shell: git add -A && git commit -m 'organize'");
    push("obs", "15 files changed, 23M committed");
    push("agent", "Committed — working tree clean.");
    // Turn 13: tighten permissions.
    push("you", "Lock down the scripts to owner-only.");
    push("think", "Removing group/other access recursively.");
    push("act", "▸ shell: chmod -R o-rwx img docs src");
    push("obs", "permissions tightened");
    push("agent", "Group and other access removed.");
    // Turn 14: archive the images.
    push("you", "Archive the images.");
    push("think", "Compressing img/ into a tarball.");
    push("act", "▸ shell: tar -czf img.tar.gz img");
    push("obs", "img.tar.gz, 11M");
    push("agent", "Archived to img.tar.gz (11M).");
    // Turn 15: verify the archive.
    push("you", "Verify the archive.");
    push("act", "▸ shell: tar -tzf img.tar.gz");
    push("obs", "4 entries ok");
    push("agent", "Archive verified — 4 entries intact.");
    // Turn 16: disk usage again.
    push("you", "Show disk usage again.");
    push("act", "▸ shell: du -sh .");
    push("obs", "34M with the archive");
    push("agent", "34M total with the archive included.");
    // Turn 17: drop the archive (confirm-gated, queued behind the lock).
    push("you", "Remove the archive, keep it lean.");
    tagQueued(2);
    push("ask", "rm img.tar.gz");
    push("act", "▸ shell: rm img.tar.gz");
    push("obs", "removed 1 file");
    push("agent", "Archive removed. Back to 23M.");
    // Turn 18: changelog entry (compaction bubble included).
    push("you", "Write a CHANGELOG entry.");
    push("think", "Appending one dated line.");
    push("act", "▸ shell: appending to CHANGELOG.md");
    push("obs", "CHANGELOG +3 lines");
    markLastAct(true);
    push("compact", "⚡ context compacted 12.4k → 3.1k tokens");
    push("agent", "**CHANGELOG.md** updated:\n- `img.tar.gz` removed\n```sh\ncat CHANGELOG.md\n```");
    // Turn 19: final sweep (timestamped thought, denied tool, error line).
    push("you", "Final check — anything left?");
    push("think", "Re-scanning one level deep.", now);
    push("act", "▸ glob: * in /current");
    push("obs", "↳ glob evidence (9 chars — agent sees it, hidden here)");
    markLastAct(true);
    push("act", "▸ shell: sudo reboot");
    markLastAct(false);
    push("deny", "✗ denied: policy (sudo)");
    push("err", "error: TTS voice unavailable — reply shown as text only");
    push("agent", "Nothing left. Tree is clean.");
    // Turn 20: user wraps up, agent closes out.
    push("you", "Great, session over.");
    push("act", "todos: ✓ List files | ✓ Group by type | ✓ Move files");
    push("agent", "All done. 40 turns shown as template.");
  }


  // The farewell card, printed once the alternate screen is gone. The name is
  // the model's own two-word title when it arrived, else an offline guess
  // from the first utterance. With nothing said at all there is no session
  // worth resuming, so nothing is printed.
  let epilogueShown = false;
  function exit() {
    if (epilogueShown) return;
    epilogueShown = true;
    const id = sid();
    renderer.destroy();
    if (firstUtterance || sessionTitle()) {
      // The "RT" half of the wordmark takes the active shade. `none` means
      // no shade at all, which is also when RT stays plain bold.
      const sh = shade();
      printEpilogue(sessionTitle() || sessionName(firstUtterance), id, sh ? shadeHex(sh, props.theme.mode) : undefined);
    }
    process.exit(0);
  }

  const calm = () => setViz({ mode: "idle", bins: new Array(NB).fill(0) });
  function clearStream() {
    streamBuf = "";
    streamDirty = false;
    setStreamText("");
  }
  function incSpeak() {
    speakCount += 1;
    speaking = true;
  }
  function decSpeak() {
    speakCount = Math.max(0, speakCount - 1);
    if (speakCount === 0) {
      speaking = false;
      lastSpeakEnd = Date.now();
    }
  }
  function setMic(next?: boolean) {
    const v = next ?? !micRef;
    micRef = v;
    setMicOn(v);
    note( v ? "mic on — press v to talk" : "mic off — voice input paused");
  }

  async function waitSilent(timeoutMs = 30000): Promise<void> {
    const t0 = Date.now();
    while ((speaking || Date.now() - lastSpeakEnd < 1200) && Date.now() - t0 < timeoutMs) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }

  // (Boot pushes nothing: theme/shade live in the left panel, and the
  // backend warm sequence reports through the status line instead.)

  // Live mic monitor: mic ON means the equalizer hears the room. One
  // arecord stays open while idle (levels only — no STT, no turns) and
  // yields while a turn records/plays so captures never fight.
  function killMon() {
    const p = micMon;
    micMon = null;
    // SIGKILL: arecord sometimes shrugs off SIGTERM and lingers.
    if (p) {
      try {
        p.kill("SIGKILL");
      } catch {
        /* dead */
      }
    }
  }
  createEffect(() => {
    const on = micOn();
    if (!on) {
      killMon();
      return;
    }
    let dead = false;
    let child: ChildProcess;
    try {
      child = spawn("arecord", ["-q", "-f", "S16_LE", "-r16000", "-c1", "-t", "raw", "-"], {
        stdio: ["ignore", "pipe", "ignore"],
      });
    } catch {
      push("err", "arecord missing — mic monitor off");
      micRef = false;
      setMicOn(false);
      return;
    }
    micMon = child;
    let last = 0;
    child.stdout?.on("data", (d: Buffer) => {
      if (dead || !micRef) return;
      if (busy() || speaking || recording) return;
      const now = Date.now();
      if (now - last < 120) return;
      last = now;
      try {
        // Room-hiss gate: per-chunk normalized spectrum dances on noise,
        // so only real levels reach the bars; silence stays flat stubs.
        const bins = rmsLevel(d) < 0.12 ? new Array(NB).fill(0) : spectrum(pcmToF32(d));
        setViz({ mode: "user", bins });
      } catch {
        /* bad chunk — skip a frame */
      }
    });
    child.on("error", () => {
      if (dead) return;
      micRef = false;
      setMicOn(false);
      push("err", "mic monitor failed — mic off");
    });
    child.on("close", () => {
      if (micMon === child) micMon = null;
    });
    onCleanup(() => {
      dead = true;
      killMon();
    });
  });

  // idle wave + stream flush (token events accumulate off-render).
  onMount(() => {
    const t = setInterval(() => {
      setTick((x) => x + 1);
      if (streamDirty) {
        streamDirty = false;
        setStreamText(streamBuf.slice(-400));
      }
    }, 120);
    onCleanup(() => clearInterval(t));
  });

  // bridge health dot
  onMount(() => {
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
    onCleanup(() => {
      stop = true;
      clearInterval(t);
    });
  });

  async function playAudio(wav: Float32Array, sr: number): Promise<void> {
    // A new playback kills any stray previous player: two overlapping
    // voices sound like reverb and double the echo surface.
    try {
      curPlayer?.kill("SIGKILL");
    } catch {
      /* ignore */
    }
    curPlayer = null;
    incSpeak();
    let mine: ChildProcess | null = null;
    try {
      const frames = envelope(wav);
      const dur = wav.length / sr;
      const child = spawn("aplay", ["-q", "-f", "S16_LE", `-r${sr}`, "-c1", "-t", "raw", "-"], {
        stdio: ["pipe", "ignore", "ignore"],
      });
      mine = child;
      curPlayer = child;
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
      if (mine && curPlayer === mine) curPlayer = null;
      decSpeak();
      calm();
    }
  }

  // Spoken replies stay SHORT (see speakHead): full outputs live in the
  // chat bubble; voicing multi-KB listings is unlistenable echo surface.
  async function speakAgent(text: string): Promise<void> {
    // Voice pipeline (TTS) runs in voice mode only; auto mode is text.
    if (mode() !== "voice") return;
    const said = speakHead(text);
    if (said.trim()) lastSpoken = [...lastSpoken.slice(-1), said];
    // Mark speaking for the WHOLE operation including TTS synthesis:
    // otherwise a v-press during the HTTP call opens the mic and the
    // reply starts mid-recording.
    incSpeak();
    try {
      // Streamed TTS: each sentence plays as soon as it lands,
      // instead of after the whole reply is synthesized.
      let any = false;
      await sayStream(said, async (wav, sr) => {
        any = true;
        await playAudio(wav, sr);
      });
      if (!any) throw new Error("tts unavailable");
    } catch (e) {
      push("err", `server TTS failed (${String(e)}) — reply shown as text only`);
    } finally {
      decSpeak();
    }
  }

  // Live loop is VOICE-mode only: auto mode is plain text+LLM with no
  // voice involved (no mic, no TTS). Voice chains listen→turn→speak→listen.
  function chainNext() {
    if (busy() || mode() !== "voice") return;
    void (async () => {
      await waitSilent();
      if (!busy() && mode() === "voice") void voiceStep(seconds, true);
    })();
  }

  // Mic primitive: records `secs` of PCM, drives the viz, returns null
  // when nothing usable was captured. While it runs: interim STT polls
  // the bridge every ~300ms into the live draft, the VAD speech
  // signal lights from chunk energy, and the dictation clock ticks.
  async function listenOnce(secs: number): Promise<Buffer | null> {
    const chunks: Buffer[] = [];
    let child: ChildProcess;
    try {
      child = spawn("arecord", ["-q", "-f", "S16_LE", "-r16000", "-c1", "-t", "raw", "-"], {
        stdio: ["ignore", "pipe", "ignore"],
      });
    } catch {
      push("err", "arecord missing — type your command instead");
      return null;
    }
    recording = true;
    setDictating(true);
    setRecSeconds(0);
    if (recTimer) clearInterval(recTimer);
    recTimer = setInterval(() => setRecSeconds((s) => s + 1), 1000);
    // Cumulative PCM for the interim poll: Whisper re-transcribes
    // the whole buffer each time, so the draft keeps improving.
    // Capped to the last 8s — same window the bridge enforces.
    let cum = Buffer.alloc(0);
    let lastInterim = "";
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    try {
      setViz({ mode: "user", bins: new Array(NB).fill(0) });
      child.stdout?.on("data", (d: Buffer) => {
        chunks.push(d);
        setViz({ mode: "user", bins: spectrum(pcmToF32(d)) });
        noteSpeech(rmsLevel(d) >= 0.12);
        cum = Buffer.concat([cum, d]);
        if (cum.length > 16000 * 8 * 2) cum = cum.slice(-16000 * 8 * 2);
      });
      pollTimer = setInterval(() => {
        if (cum.length < 1600) return;
        void sttInterim(cum)
          .then((r) => {
            if (r.kind === "text" && r.text && r.text !== lastInterim) {
              lastInterim = r.text;
              setDraft(r.text);
            }
          })
          .catch(() => {});
      }, 300);
      await new Promise((r) => setTimeout(r, secs * 1000));
      child.kill("SIGTERM");
      calm();
      const pcm = Buffer.concat(chunks);
      return pcm.length >= 3200 ? pcm : null;
    } finally {
      recording = false;
      setDictating(false);
      if (recTimer) {
        clearInterval(recTimer);
        recTimer = null;
      }
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      setSpeech(false);
      if (speechTimer) {
        clearTimeout(speechTimer);
        speechTimer = null;
      }
      // Draft stays set: runText clears it once the turn runs.
    }
  }

  // One hands-free voice step: wait for our own playback to end, listen,
  // STT, discard self-echo, run turn (chains again at summary in loops).
  // `force` bypasses the busy gate because a summary means the previous
  // turn already ended (ref updates lag state).
  async function voiceStep(secs: number, force = false): Promise<void> {
    if (mode() !== "voice") return;
    if (!force && busy()) return;
    if (!micRef) {
      setBusy(false);
      if (!force) note( "mic is off — press m to enable.");
      return;
    }
    await waitSilent();
    if (!force && busy()) return;
    setBusy(true);
    // Chained (force) steps keep the live loop going through silence,
    // capped so a dead room doesn't spin forever.
    const keepAlive = () => {
      setBusy(false);
      if (force) {
        quietStreak += 1;
        if (quietStreak > 12) {
          quietStreak = 0;
          note( "quiet for a while — Tab to resume the live loop.");
          return;
        }
        chainNext();
      }
    };
    note( "recording — speak now");
    const pcm = await listenOnce(secs);
    if (!pcm) {
      note( "nothing recorded.");
      keepAlive();
      return;
    }
    try {
      const r = await stt(pcm);
      if (r.kind !== "text" || !r.text?.trim()) {
        note( `STT empty (${r.message ?? "silence?"})`);
        keepAlive();
        return;
      }
      // Echo guard: the mic may still catch our speaker (no headphones).
      // Matches full or partial re-hearings of our last replies.
      if (isEcho(lastSpoken, r.text)) {
        note( "heard my own voice — discarded.");
        keepAlive();
        return;
      }
      quietStreak = 0;
      setBusy(false);
      runText(r.text);
    } catch (e) {
      push("err", `STT failed: ${String(e)}`);
      setBusy(false);
    }
  }

  // Voice-mode confirm gate: speak the question, listen for yes/no.
  async function voiceConfirm(action: Record<string, unknown>): Promise<void> {
    if (!micRef) {
      note( "mic is off — confirm denied. Press m to enable.");
        confirmSock(false);
      return;
    }
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
        note( `heard: "${r.text ?? ""}" → ${ok ? "confirmed ✓" : "denied."}`);
      } catch {
        note( "confirm listen failed → denied.");
      }
    } else {
      note( "no answer → denied.");
    }
    confirmSock(ok);
  }

  function runText(text: string) {
    // Voice turns reach here via STT, typed ones via onSubmit; either way
    // this is the first moment we know what the user actually asked for.
    firstUtterance ||= text.trim();
    if (busy() || !sock?.connected) {
      if (!sock?.connected) push("err", "not connected to bridge yet");
      return;
    }
    setBusy(true);
    turnChatCount = 0;
    clearStream();
    setHistOff(0);
    setPhase("thinking …");
    setDraft("");
    push("you", text);
    sock.turn(text, sid(), cwd());
  }

  // persistent socket — autonomous DeepAgent (MCP + todos) is the main,
  // with TermSocket as fallback. Both speak the same event protocol.
  onMount(() => {
    // Boot guard: without it a re-run effect (or a remount) opens a
    // SECOND live socket and every turn renders twice.
    if (booted) return;
    booted = true;
    let active: DeepSocket | TermSocket | null = null;
    let cancelled = false;

    const setupHandlers = (sockInst: DeepSocket | TermSocket) => {
      sockInst.onEvent = (e: TurnEvent) => {
        if (e.event === "route") {
          // Which brain took the turn (two-brain delegation). Chat turns
          // answer on the small front model and never reach the worker, so
          // this is the only place that fact is visible.
          const r = e as { brain?: string; kind?: string; reason?: string; forced?: boolean };
          const brain = r.brain === "front" ? "front" : "worker";
          const why = r.forced ? "forced" : r.reason === "backstop" ? "backstop" : "";
          setPhase(brain === "front" ? "front model …" : "worker …");
          if (brain === "worker") note( `→ worker${why ? ` (${why})` : ""}`);
          // The turn actually started: clear any queued badge.
          setMsgs((m) => {
            const i = m.map((x) => x.who).lastIndexOf("you");
            if (i < 0 || m[i]!.queued === undefined) return m;
            const copy = m[i]!;
            return [...m.slice(0, i), { ...copy, queued: undefined }, ...m.slice(i + 1)];
          });
        } else if (e.event === "queued") {
          // The session lock is held: our text waits. Tag the last
          // user bubble with its position so input never vanishes
          // silently (it was queued server-side, not dropped).
          const pos = Number((e as { position?: number }).position ?? 1) || 1;
          setMsgs((m) => {
            const i = m.map((x) => x.who).lastIndexOf("you");
            if (i < 0) return m;
            const copy = m[i]!;
            return [...m.slice(0, i), { ...copy, queued: pos }, ...m.slice(i + 1)];
          });
          setPhase(`queued #${pos} …`);
        } else if (e.event === "action") {
          clearStream();
          const a = (e as { action: Record<string, unknown> }).action ?? {};
          lastOp = String(a["action"] ?? "");
          setPhase("toolcall …");
          if (String(a["action"]) === "write_todos") {
            const list = (a["todos"] as unknown as Array<{ content: string; status: string }>) || [];
            if (Array.isArray(list) && list.length) {
              setTodos(list.slice(0, 8));
              push("act", `todos: ${list.map((t) => `${t.status === "completed" ? "✓" : t.status === "in_progress" ? "●" : "○"} ${t.content}`).join(" | ")}`);
            }
            return;
          }
          push("act", `▸ ${String(a["action"] ?? "?")}: ${String(a["command"] ?? a["path"] ?? a["pattern"] ?? "")}`);
        } else if (e.event === "observation") {
          clearStream();
          // Good tool call sign: the action that produced this
          // observation is done — mark it ✓ in place.
          setMsgs((m) => {
            for (let i = m.length - 1; i >= 0; i--) {
              if (m[i]!.who === "act") {
                if (m[i]!.done) break;
                const copy = m[i]!;
                return [...m.slice(0, i), { ...copy, done: true }, ...m.slice(i + 1)];
              }
              if (m[i]!.who === "you") break;
            }
            return m;
          });
          const text = String((e as { observation: string }).observation ?? "").slice(0, 800);
          // Web evidence stays agent-side: the dump floods the screen and
          // the agent already consumed it. Everything else renders capped.
          const op = lastOp;
          if (op === "web_search" || op === "searxng" || op === "fetch") {
            push("obs", `↳ ${op} evidence (${text.length} chars — agent sees it, hidden here)`);
          } else {
            push("obs", text);
          }
        } else if (e.event === "deny") {
          // Policy block (or user cancel): the tool call did NOT
          // run. Red ✗ line — previously these were invisible.
          clearStream();
          setMsgs((m) => {
            for (let i = m.length - 1; i >= 0; i--) {
              if (m[i]!.who === "act") {
                if (m[i]!.done) break;
                const copy = m[i]!;
                return [...m.slice(0, i), { ...copy, done: false }, ...m.slice(i + 1)];
              }
              if (m[i]!.who === "you") break;
            }
            return m;
          });
          push("deny", `✗ denied: ${String((e as { reason?: string }).reason ?? "policy")}`);
        } else if (e.event === "chat") {
          clearStream();
          setPhase("answering …");
          const reply = String((e as { reply: string }).reply ?? "");
          if (reply.trim()) {
            turnChatCount += 1;
            push("agent", reply);
            void speakAgent(reply);
          }
        } else if (e.event === "thinking") {
          const th = String((e as { text: string }).text ?? "").slice(0, 1200);
          const append = (e as { append?: boolean }).append === true;
          const ts = (e as { ts?: number }).ts;
          if (th) {
            setPhase("thinking …");
            if (append) {
              // Live think lane: grow the last think bubble instead of
              // spraying one bubble per delta (dozens per step).
              setMsgs((m) => {
                const last = m[m.length - 1];
                if (last && last.who === "think") {
                  const grown = (last.text + th).slice(-800);
                  if (grown === last.text && (last.ts ?? 0) === (ts ?? 0)) return m;
                  return [...m.slice(0, -1), { ...last, text: grown, ts: ts ?? last.ts }];
                }
                const capped = capText("think", th);
                if (!capped.trim()) return m;
                return [...m.slice(-199), { id: uid(), who: "think" as const, text: capped, ts }];
              });
            } else push("think", th, ts);
          }
        } else if (e.event === "cancelled") {
          // Our own ctrl+c landed mid-turn: say so instead of
          // leaving the phase spinner running forever.
          clearStream();
          setBusy(false);
          setPhase("interrupted");
          note( "↩ turn interrupted");
        } else if (e.event === "token") {
          // Live LLM chips accumulate off-render; the tick flushes them to
          // the streaming agent bubble below. Raw pieces may include think
          // tags mid-stream — the final parsed events replace this preview.
          // Phase flips to LLM so the state line reads Thinking → LLM →
          // Tool call → Answering instead of sticking on Thinking.
          const piece = String((e as { piece: string }).piece ?? "");
          if (piece) {
            streamBuf = (streamBuf + piece).slice(-1200);
            streamDirty = true;
            setPhase((p) => (p === "toolcall …" ? p : "LLM …"));
          }
        } else if (e.event === "compact" || e.event === "compaction") {
          const note = String((e as { note?: string; message?: string }).note ?? (e as { note?: string; message?: string }).message ?? "context compacted");
          push("compact", note);
        } else if (e.event === "stuck") {
          note( `↻ stuck: ${String((e as { reason: string }).reason ?? "")}`);
        } else if (e.event === "audio") {
          if (mode() === "voice") {
            const ev = e as { wav_b64: string; sr: number };
            void playAudio(b64ToF32(ev.wav_b64), ev.sr ?? 24000);
          }
        } else if (e.event === "summary") {
          // Fallback visibility: if the turn ran but produced no chat
          // bubble (e.g. tools-only finish), surface the summary reply
          // so the assistant never looks silent.
          const reply = String((e as { reply: string }).reply ?? "");
          if (turnChatCount === 0 && reply.trim()) {
            turnChatCount += 1;
            push("agent", reply);
            void speakAgent(reply);
          }
          clearStream();
          setBusy(false);
          setPhase("");
          chainNext();
        } else if (e.event === "title") {
          // Model-named the session (bridge sends this once, after turn 1).
          // Silent by design: it is for the exit card, not the transcript.
          const t2 = String((e as { title?: string }).title ?? "").trim();
          if (t2) setSessionTitle(t2);
        } else if (e.event === "error") {
          clearStream();
          setPhase("");
          push("err", `error: ${String((e as { message: string }).message ?? "")}`);
          setBusy(false);
        }
      };
      // Both sockets carry the confirm gate (the autonomous socket
      // runs the same terminal tools as the fallback one).
      sockInst.onConfirm = (action) => {
        if (mode() === "voice") void voiceConfirm(action);
        else setPending(action);
      };
      sock = sockInst;
      active = sockInst;
    };

    // Single local app: reuse a healthy backend if one is already up,
    // else spawn bridge.py as our own child (killed with us on exit).
    // Warm chatter goes to the left dashboard (backendNote), not the chat.
    const boot = async () => {
      const want = await ensureBackend((m) => setBackendNote(m));
      if (cancelled) return;
      if (!want) {
        setBackendNote("failed to start");
        push("err", "local agent backend failed to start (see terminal above)");
        return;
      }
      try {
        const mi = await getModel();
        const cur = mi.available.find((a) => a.current) ?? mi.available[0];
        if (cur?.label) setModelLabel(cur.label);
      } catch {
        /* keep fallback label */
      }
      // A resumed session already has a name on disk. Fetching it here is
      // what makes `voicert -s <id>` then an immediate quit still print the
      // original title — no turn has run, so no title event is coming.
      void fetchTitle(sid()).then((t) => {
        if (t && !cancelled) setSessionTitle(t);
      });
      // Now that backend is up, connect the autonomous socket
      let sockInst: DeepSocket | TermSocket = new DeepSocket();
      setupHandlers(sockInst);
      try {
        await (sockInst as DeepSocket).connect();
        active = sockInst;
        // Warmed + connected: the dashboard falls back to its steady
        // "● agent ready" (metrics-driven) from here on.
        setBackendNote("");
      } catch {
        const fallback = new TermSocket();
        setupHandlers(fallback);
        try {
          await fallback.connect();
          active = fallback;
          setBackendNote("");
        } catch {
          setBackendNote("socket failed — retrying");
          push("err", "bridge unreachable after start — retrying on next turn");
        }
      }
    };
    void boot();

    onCleanup(() => {
      cancelled = true;
      try {
        // Close the socket so a stale connection can never keep pushing
        // duplicate events after unmount/remount.
        active?.close();
      } catch {
        /* ignore */
      }
      if (active && sock === active) sock = null;
      stopBackend();
    });
  });

  async function onSubmit(text: string): Promise<void> {
    setValue("");
    setCmdSel(0);
    setCmdDismissed(false);
    const t = text.trim();
    if (!t || busy()) return;
    // Remember the first real utterance for the farewell card. Done here,
    // before the slash-command switch and before the bridge check, so the
    // name reflects what the user asked for even if the turn then failed.
    if (!t.startsWith("/")) firstUtterance ||= t;
    if (t === "/quit" || t === "/q") {
      exit();
      return;
    }
    if (t === "/new") {
      // A new session has no name and no history: clear the title too, or
      // /new would keep printing the old session's name on exit.
      firstUtterance = "";
      setSessionTitle("");
      setSid(newSessionId());
      setHistOff(0);
      note( "new session");
      return;
    }
    if (t === "/clear") {
      setMsgs([]);
      setHistOff(0);
      setStatus("");
      return;
    }
    if (t === "/cwd" || t.startsWith("/cwd ")) {
      const p = t.split(/\s+/, 2)[1];
      // Bare /cwd reports where we are instead of silently doing nothing.
      if (!p) {
        note( `cwd ${cwd()}`);
        return;
      }
      setCwd(p);
      note( `cwd ${p}`);
      return;
    }
    if (t.startsWith("/voice")) {
      if (mode() !== "voice") {
        note( "voice lives in voice mode — Tab to switch.");
        return;
      }
      const s = Number(t.split(/\s+/, 2)[1]) || seconds;
      void voiceStep(s);
      return;
    }
    if (t === "/template") {
      seedTemplate();
      return;
    }
    if (t === "/panel") {
      setShowPanel((v) => {
        const n = !v;
        note(`panel ${n ? "shown" : "hidden"}`);
        return n;
      });
      return;
    }
    if (t === "/mode") {
      note( `mode: ${mode()} (Tab toggles auto ↔ voice)`);
      return;
    }
    if (t === "/opencode" || t.startsWith("/opencode ")) {
      const arg = t.slice("/opencode".length).trim();
      const targetCwd = arg || cwd();
      note( `spawning opencode in ${targetCwd}…`);
      const { spawnOpencode } = await import("./opencode.js");
      const ok = await spawnOpencode(targetCwd, (m) => note( m));
      if (!ok) push("err", "opencode spawn failed — is kitty/tmux installed?");
      return;
    }
    if (t.startsWith("/mode ")) {
      const m = t.split(/\s+/, 2)[1] as Mode;
      if (m === "auto" || m === "voice") {
        setMode(m);
        note( `mode → ${m}`);
      } else push("err", "mode must be auto|voice");
      return;
    }
    if (t === "/model") {
      try {
        const mi = await getModel();
        // Verbose dumps stay in the transcript (as neutral lines): they are
        // answers the user explicitly asked for, not status noise.
        push("obs", mi.available.map((a) =>
          `${a.current ? "●" : "○"} ${a.name} — ${a.label} [${a.backend}]${a.ready ? "" : ` (blocked: ${a.desc})`}`).join("\n"));
      } catch {
        push("err", "could not reach /model (bridge down?)");
      }
      return;
    }
    if (t.startsWith("/model ")) {
      const name = (t.split(/\s+/, 2)[1] ?? "").trim();
      if (!name) {
        push("err", "usage: /model <name> (see /model)");
        return;
      }
      note( `switching model → ${name} (re-warming LLM leg…)`);
      try {
        const r = await switchModel(name);
        if (r.kind === "ok") {
          if (r.label) setModelLabel(r.label);
          if (r.compacted) push("compact", r.compacted);
          note( `model → ${r.label ?? r.current ?? name}${r.note ? ` (${r.note})` : ""}${r.compacted ? ` · context ${r.compacted}` : ""}`);
        } else push("err", `model switch failed: ${r.message ?? "unknown"}`);
      } catch (e) {
        push("err", `model switch failed: ${String(e)}`);
      }
      return;
    }
    if (t === "/settings") {
      // Full legs + policy dump: the dashboard panel shows the
      // short tags; this is the verbose form.
      try {
        const legs = await getLegs();
        push("obs",
          [
            `model: ${legs.model.label} (${legs.model.name} · ${legs.model.backend})`,
            `vad: ${legs.legs.vad}`,
            `stt: ${legs.legs.stt}`,
            `llm: ${legs.legs.llm}`,
            `tts: ${legs.legs.tts}`,
            `ops: ${legs.policy.ops.join(" ")}`,
            `deny: ${legs.policy.deny.join(" ")}`,
            `confirm: ${legs.policy.confirm.join(" ")}`,
          ].join("\n"));
      } catch {
        push("err", "could not reach /legs (bridge down?)");
      }
      return;
    }
    if (t === "/dictate" || t.startsWith("/dictate ")) {
      const s = Number(t.split(/\s+/, 2)[1]) || seconds;
      note( "dictation listening — speak now…");
      void (async () => {
        const pcm = await listenOnce(s);
        if (pcm) {
          try {
            const r = await stt(pcm);
            const txt = r.text?.trim();
            if (r.kind === "text" && txt) {
              note( `dictated: "${txt}"`);
              setValue((v) => (v ? `${v} ${txt}` : txt));
            }
          } catch (e) {
            push("err", `dictation STT failed: ${String(e)}`);
          }
        }
      })();
      return;
    }
    if (t === "/mic" || t.startsWith("/mic ")) {
      // Any argument must be on/off. Matching only the two exact strings let
      // "/mic bogus" fall through to the unknown-command handler and report
      // "did you mean /mic?", hiding the real fix.
      const arg = t.slice(4).trim().toLowerCase();
      if (arg && arg !== "on" && arg !== "off") {
        push("err", "usage: /mic [on|off]");
      } else {
        setMic(arg === "on" ? true : arg === "off" ? false : undefined);
      }
      return;
    }
    if (t === "/shade" || t.startsWith("/shade ")) {
      const arg = t.slice("/shade".length).trim().toLowerCase();
      if (!arg) {
        const cur = shade() ?? "none";
        note( `shade: ${cur} · ${SHADES.map((s) => (s === cur ? `● ${s}` : `○ ${s}`)).join(" ")} · none = unshaded`);
        return;
      }
      if (arg === "none" || arg === "off") {
        setShade(null);
        note( "shade → none (default palette)");
        return;
      }
      if (!isShade(arg)) {
        push("err", `usage: /shade <${SHADES.join("|")}|none>`);
        return;
      }
      setShade(arg);
      note( `shade → ${arg} (monotonic; backdrop unchanged)`);
      return;
    }
    if (t === "/help") {
      // Built from COMMANDS so the menu and /help can never disagree.
      const names = COMMANDS.map((c) => c.name).join(" ");
      push("obs", `${names}\nType / for the menu · Tab completes · ↑↓ pick · enter runs`);
      return;
    }
    if (t.startsWith("/")) {
      // Suggest the closest name instead of a bare "unknown" — the palette is
      // closed by now (an argument was typed), so this is the only guidance.
      // Prefix-only matching missed plain typos: "/modle" matched nothing even
      // though "/mode" is one transposition away, so allow a small edit budget.
      const tok = t.slice(1).split(/\s/, 1)[0].toLowerCase();
      const near = COMMANDS.map((c) => ({ name: c.name, d: editDistance(tok, c.name.slice(1)) }))
        .filter((c) => c.d <= Math.max(1, Math.floor(c.name.length / 3)))
        .sort((a, b) => a.d - b.d)
        .map((c) => c.name);
      push("err", near.length ? `unknown ${t} — did you mean ${near.slice(0, 2).join(" or ")}?` : `unknown ${t} — /help`);
      return;
    }
    runText(t);
  }

  // OpenTUI types the input's onSubmit as both (value) and (SubmitEvent);
  // accept either and fall back to the controlled value.
  function submitFromInput(v: unknown): void {
    void onSubmit(typeof v === "string" ? v : value());
  }

  useKeyboard((key) => {
    // Terminals disagree on what a modified key is called: the legacy
    // parser lowercases letters and reports ctrl as a flag, but under the
    // kitty protocol Ctrl+O arrives as the raw control codepoint (0x0f)
    // and a shifted "Y" keeps its uppercase name. Normalize both.
    const name = key.name.toLowerCase();
    const ctrlName =
      key.ctrl && key.name.length === 1 && key.name.charCodeAt(0) < 32
        ? String.fromCharCode(key.name.charCodeAt(0) + 96)
        : name;
    if (key.ctrl && ctrlName === "c") {
      key.preventDefault();
      // Selection wins over everything: Shift+drag selects in the chat
      // panel, Ctrl+C copies it (OSC52) instead of interrupting/exiting.
      try {
        const r = renderer as unknown as {
          hasSelection?: boolean;
          getSelection?: () => { getSelectedText?: () => string } | null;
          copyToClipboardOSC52?: (t: string) => boolean;
          clearSelection?: () => void;
        };
        if (r.hasSelection) {
          const text = r.getSelection?.()?.getSelectedText?.() ?? "";
          if (text.trim()) {
            r.copyToClipboardOSC52?.(text);
            r.clearSelection?.();
            note(`copied ${text.length} chars to clipboard`);
            return;
          }
        }
      } catch {
        /* no selection support — fall through to interrupt/exit */
      }
      // Busy turn: interrupt it (bridge cancels the task) instead of
      // quitting the whole app. Idle: the classic exit.
      if (busy()) {
        sock?.cancel();
        note( "interrupting…");
      } else exit();
      return;
    }
    if (name === "pageup") {
      scrollToPrevUser();
      key.preventDefault();
      return;
    }
    if (name === "pagedown") {
      scrollToNextUser();
      key.preventDefault();
      return;
    }
    if (name === "tab") {
      // Tab completes the highlighted command when the palette is open, and
      // otherwise cycles modes everywhere (also while typing — Tab never edits).
      key.preventDefault();
      const pick = cmdOpen() ? cmdVisible()[cmdIndex()] : undefined;
      if (pick) {
        setValue(`${pick.name} `);
        setCmdSel(0);
        setCmdDismissed(true);
        return;
      }
      setMode((m) => {
        const n = MODES[(MODES.indexOf(m) + 1) % MODES.length] ?? "auto";
        note( `mode → ${n}${n === "voice" ? " (live voice pipeline)" : " (text + LLM, no voice)"}`);
        return n;
      });
      return;
    }
    // Palette navigation: arrows move the highlight, enter runs it. Bound only
    // while the menu is open so plain-text editing keeps the arrows.
    if (cmdOpen() && (name === "up" || name === "down")) {
      key.preventDefault();
      const n = cmdMatches().length;
      const step = name === "down" ? 1 : -1;
      setCmdSel((s) => (s + step + n) % n);
      return;
    }
    if (cmdOpen() && name === "return") {
      const pick = cmdVisible()[cmdIndex()];
      // Only take over Enter when the highlight was actually moved off the top
      // row, or when the text is not already an exact command. Otherwise enter
      // on a typed "/mode" ran the top match "/model" instead — submitting a
      // prefix silently executed a different command than the one on screen.
      // Arrows move the highlight, so Enter runs it. Tab-less Enter on an
      // untouched menu just submits the text as typed.
      const typed = value().trim();
      const exact = COMMANDS.some((c) => c.name === typed);
      if (pick && !exact && cmdIndex() !== 0) {
        key.preventDefault();
        setValue("");
        setCmdSel(0);
        setCmdDismissed(false);
        void onSubmit(pick.name);
        return;
      }
    }
    const pend = pending();
    if (pend) {
      if (name === "y") {
        key.preventDefault();
        note( "confirmed ✓");
        setPending(null);
        confirmSock(true);
      } else if (name === "n" || name === "escape") {
        key.preventDefault();
        note( "denied.");
        setPending(null);
        confirmSock(false);
      }
      return;
    }
    if (name === "escape") {
      // Keep the prompt focused: opencode's input blurs on escape. With the
      // palette up, escape closes the menu first and leaves the text alone.
      key.preventDefault();
      if (cmdOpen()) setCmdDismissed(true);
      return;
    }
    if (key.ctrl && ctrlName === "o") {
      key.preventDefault();
      const targetCwd = cwd();
      note( `spawning opencode in ${targetCwd}…`);
      void import("./opencode.js").then(({ spawnOpencode }) => spawnOpencode(targetCwd, (m) => note( m)));
      return;
    }
    if (key.shift && (name === "return" || name === "enter")) {
      key.preventDefault();
      setValue((v) => v + "\n");
      return;
    }
    // Typing into the prompt must never trigger the bare-letter shortcuts.
    if (name === "m" && value().trim() === "" && !busy()) {
      key.preventDefault();
      setMic();
      return;
    }
    if (name === "d" && value().trim() === "" && !busy()) {
      key.preventDefault();
      void onSubmit("/dictate");
      return;
    }
    if (name === "v" && value().trim() === "" && !busy()) {
      key.preventDefault();
      if (mode() !== "voice") {
        note( "voice lives in voice mode — Tab to switch.");
        return;
      }
      void voiceStep(seconds);
    }
  });

  // --- derived view state -------------------------------------------------
  function fmtTime(ts?: number): string {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    return d.toTimeString().slice(0, 8);
  }

  const inputLines = () => {
    const v = value();
    if (!v) return 1;
    const availW = Math.max(20, termCols() - leftW() - 16);
    const parts = v.split("\n");
    let total = 0;
    for (const part of parts) {
      total += Math.max(1, Math.ceil((part.length || 1) / availW));
    }
    // Grows with Shift+Enter newlines and with wrapping past the wall.
    return Math.min(8, Math.max(1, total));
  };

  const colorFor = (w: Msg["who"]) =>
    w === "you"
      ? theme().accent
      : w === "agent"
      ? theme().ink
      : w === "err"
      ? theme().danger
      : w === "ask"
      ? theme().warn
      : w === "compact"
      ? theme().warn
      : theme().dim;

  const labelFor = (w: Msg["who"]) =>
    w === "you" ? "› " : w === "ask" ? "⬡ Confirm? " : w === "think" ? "› think " : "";

  // Visible transcript window, measured in terminal ROWS rather than messages:
  // a bordered user turn costs 3 rows (top border + text + bottom border)
  // while plain lines cost wrapped-text rows. The right column has a fixed
  // height, and overflowing it makes Yoga shrink every child so rows paint
  // on top of each other — so the window must fit EXACTLY into the space
  // left by the chrome (panel borders, phase, palette, input).
  // Two modes: live (histOff 0) fills newest-first so the latest turn and the
  // input stay visible; scrolled (histOff > 0) is TOP-ANCHORED at a user
  // message and fills forward, so a scrolled view always opens on the turn's
  // input with the assistant's reply below it. histOff is the message count
  // from the end to the top line (len - topIndex).
  // Chrome rows: 2 panel borders + 1 phase + input box, plus the transient
  // stream / status / palette / confirm lines when they are on screen.
  const rowBudget = () => {
    const inputH = inputLines() > 1 ? inputLines() + 2 : 3;
    // Streaming bubbles eat transcript rows too: estimate from wrapped
    // length so a long LLM preview can't push the input off-screen.
    const innerW = Math.max(20, termCols() - leftW() - 10);
    const streamRows = streamText()
      ? Math.min(6, Math.max(1, Math.ceil(streamText().length / innerW)))
      : 0;
    const draftRows = dictating() || draft()
      ? Math.min(4, Math.max(3, Math.ceil(((draft() || "").length + 8) / innerW) + 2))
      : 0;
    const chrome =
      2 + 1 + inputH +
      streamRows +
      draftRows +
      (status() ? 1 : 0) +
      (pending() ? 1 : 0) +
      (cmdOpen() ? cmdVisible().length + 2 : 0);
    // -1 safety row: wrapping is estimated, and one spare row degrades to a
    // blank line while one row over degrades to overlapping text.
    return Math.max(1, termRows() - chrome - 1);
  };
  const rowsOf = (m: Msg) => {
    const innerW = Math.max(20, termCols() - leftW() - 10);
    const textLen = (labelFor(m.who) + m.text).length;
    const lines = m.text.split("\n").length;
    return Math.max(lines, Math.ceil(textLen / innerW)) + (m.who === "you" ? 2 : 0);
  };
  const histVis = () => {
    const all = msgs();
    const budget = rowBudget();
    if (histOff() === 0) {
      let rows = 0;
      let start = all.length;
      for (let i = all.length - 1; i >= 0; i--) {
        rows += rowsOf(all[i]);
        if (rows > budget) {
          start = i + 1;
          break;
        }
        start = i;
      }
      return all.slice(start);
    }
    const top = Math.max(0, Math.min(all.length, all.length - histOff()));
    const out: Msg[] = [];
    let rows = 0;
    for (let i = top; i < all.length; i++) {
      const r = rowsOf(all[i]);
      if (rows + r > budget) break;
      rows += r;
      out.push(all[i]);
    }
    return out;
  };
  // Turn-anchored scrolling: PgUp/wheel-up jumps to the previous user input,
  // PgDn/wheel-down to the next one (or back to live past the last turn).
  const prevUserIdx = (before: number) => {
    const a = msgs();
    for (let i = Math.min(before, a.length - 1); i >= 0; i--) {
      if (a[i].who === "you") return i;
    }
    return -1;
  };
  const nextUserIdx = (after: number) => {
    const a = msgs();
    for (let i = after; i < a.length; i++) {
      if (a[i].who === "you") return i;
    }
    return -1;
  };
  function scrollToPrevUser() {
    const a = msgs();
    const cur = histOff() === 0 ? a.length : Math.max(0, a.length - histOff());
    const t = prevUserIdx(cur - 1);
    setHistOff(a.length - (t < 0 ? 0 : t));
  }
  function scrollToNextUser() {
    const a = msgs();
    if (histOff() === 0) return;
    const top = Math.max(0, a.length - histOff());
    const t = nextUserIdx(top + 1);
    setHistOff(t < 0 ? 0 : a.length - t);
  }

  // --- command palette ---------------------------------------------------
  // The typed prefix of a command, or null when the input isn't a command.
  // Only the token up to the first space counts: once an argument is being
  // typed the name is settled and the menu has nothing left to narrow.
  const cmdToken = () => {
    const v = value();
    if (!v.startsWith("/")) return null;
    if (/\s/.test(v)) return null;
    return v.slice(1).toLowerCase();
  };
  const cmdMatches = createMemo(() => {
    const tok = cmdToken();
    if (tok === null) return [];
    return COMMANDS.filter((c) => c.name.slice(1).startsWith(tok));
  });
  const cmdOpen = () => cmdMatches().length > 0 && !cmdDismissed();
  // Keep the highlight in range as the filter narrows, without an effect.
  const cmdIndex = () => Math.min(cmdSel(), Math.max(0, cmdMatches().length - 1));
  /** Highest number of rows the palette may take from the transcript. */
  const CMD_ROWS = 8;
  const cmdVisible = () => cmdMatches().slice(0, CMD_ROWS);

  return (
    <box flexDirection="row" height={termRows()} backgroundColor={theme().screen}>
      <Show when={showPanel()}>
      <box
        flexDirection="column"
        width="22%"
        flexShrink={0}
        height={termRows()}
        justifyContent="flex-start"
        backgroundColor={theme().panel}
      >
        {/* Mic widget (own width, never touches the chat): squared circle
            cell + bold status on row one, live VAD dot + voice bar on row
            two. The cell shares the widget's left wall and adds only its
            right wall, so it reads as one squared box. */}
        <box
          flexShrink={0}
          border
          borderStyle="heavy"
          borderColor={theme().border}
          flexDirection="column"
        >
          <box flexDirection="row" flexShrink={0}>
            <box
              flexShrink={0}
              border={["right"]}
              borderStyle="heavy"
              borderColor={theme().border}
              backgroundColor="#000000"
            >
              <text fg={micOn() ? rtColor() : theme().dim} attributes={TextAttributes.BOLD}>
                {"⬤"}
              </text>
            </box>
            <text fg={theme().ink}>{" "}</text>
            <text fg={theme().ink} attributes={TextAttributes.BOLD}>
              {`MIC STATUS : ${micOn() ? "ON" : "OFF"}`}
            </text>
          </box>
          <box flexDirection="row" flexShrink={0}>
            <text fg={vadLive() ? rtColor() : theme().dim} attributes={TextAttributes.BOLD}>
              {vadLive() ? "● " : "○ "}
            </text>
            <text fg={theme().accent} attributes={TextAttributes.BOLD}>
              {"█".repeat(barW())}
            </text>
          </box>
        </box>
        <Panel
          theme={theme}
          shade={shade}
          sid={sid}
          cwd={cwd}
          mode={mode}
          micOn={micOn}
          speech={speech}
          dictating={dictating}
          recSeconds={recSeconds}
          todos={todos}
          viz={viz}
          tick={tick}
          busy={busy}
          backendNote={backendNote}
        />
      </box>
      </Show>
      <box
        flexDirection="column"
        flexGrow={1}
        flexShrink={1}
        border
        borderStyle="heavy"
        borderColor={theme().border}
        paddingLeft={1}
        paddingRight={1}
        height={termRows()}
        onMouseScroll={(e: MouseEvent) => {
          // Wheel over the chat panel scrolls turn by turn, like PgUp/PgDn.
          // Needs useMouse tracking (see index.tsx).
          const dir = e.scroll?.direction ?? (e.button === 64 ? "up" : e.button === 65 ? "down" : undefined);
          if (dir === "up") scrollToPrevUser();
          else if (dir === "down") scrollToNextUser();
        }}
      >
        {/* Transcript owns the leftover space but can never push the input
            off-screen: it clips instead. histVis already windows newest-first
            by rows, so clipping is only a backstop for the open palette.
            Empty: the home screen — wordmark (RT in the live shade) plus the
            rules/suggestions list. Borderless by design: panel + input stay
            the only two heavy frames. */}
        <box flexDirection="column" flexGrow={1} overflow="hidden">
          <Show when={msgs().length === 0}>
            <box flexDirection="column" flexGrow={1} justifyContent="center" alignItems="center">
              {/* Original wordmark, 2 rows: "Voice" in terminal ink, "RT"
                  in the live shade — same split as the farewell card. */}
              <box flexDirection="row" flexShrink={0}>
                <text fg={theme().ink} attributes={TextAttributes.BOLD}>
                  {WORDMARK[0]?.slice(0, RT_COL)}
                </text>
                <text fg={rtColor()} attributes={TextAttributes.BOLD}>
                  {` ${WORDMARK[0]?.slice(RT_COL)}`}
                </text>
              </box>
              <box flexDirection="row" flexShrink={0}>
                <text fg={theme().ink} attributes={TextAttributes.BOLD}>
                  {WORDMARK[1]?.slice(0, RT_COL)}
                </text>
                <text fg={rtColor()} attributes={TextAttributes.BOLD}>
                  {` ${WORDMARK[1]?.slice(RT_COL)}`}
                </text>
              </box>
              <text fg={theme().dim}>{" "}</text>
              <For each={SUGGESTIONS}>
                {(s) => (
                  <text fg={theme().dim}>
                    {s}
                  </text>
                )}
              </For>
            </box>
          </Show>
          <For each={histVis()}>
            {(m) => (
              <>
                <Show when={m.who === "you"}>
                  <box border borderStyle="heavy" borderColor={theme().inputBorder} opacity={0.6}>
                    <box flexDirection="row" justifyContent="space-between">
                      <text fg={colorFor(m.who)}>
                        {labelFor(m.who) + m.text}
                      </text>
                      <Show when={m.queued !== undefined}>
                        <text fg={theme().warn} attributes={TextAttributes.BOLD}>
                          {` [queued #${m.queued}]`}
                        </text>
                      </Show>
                    </box>
                  </box>
                </Show>
                <Show when={m.who === "agent"}>
                  <markdown content={m.text} syntaxStyle={mdStyle} streaming={false} fg={theme().ink} />
                </Show>
                <Show when={m.who === "act"}>
                  <text
                    fg={m.done === true ? theme().accent : m.done === false ? theme().danger : theme().warn}
                    attributes={m.done === true ? TextAttributes.BOLD : TextAttributes.NONE}
                  >
                    {m.text.startsWith("todos:")
                      ? m.text
                      : `${m.done === true ? "✓ " : m.done === false ? "✗ " : "▸ "}${m.text.replace(/^[▸✓✗\s]*/, "")}`}
                  </text>
                </Show>
                <Show when={m.who === "think"}>
                  <text fg={colorFor(m.who)} attributes={TextAttributes.DIM}>
                    {`› think ${m.ts ? `[${fmtTime(m.ts)}] ` : ""}${m.text}`}
                  </text>
                </Show>
                <Show when={m.who === "compact"}>
                  <text fg={theme().warn} attributes={TextAttributes.BOLD}>
                    {`⚡ ${m.text}`}
                  </text>
                </Show>
                <Show when={m.who !== "you" && m.who !== "agent" && m.who !== "act" && m.who !== "think" && m.who !== "compact"}>
                  <text fg={colorFor(m.who)}>
                    {labelFor(m.who) + m.text}
                  </text>
                </Show>
              </>
            )}
          </For>
        </box>
        {/* Realtime STT draft: live user bubble while speaking — interim
            tokens land here token-wise, then the final STT replaces it with
            the bordered `you` message above (see runText). */}
        <Show when={dictating() || draft()}>
          <box border borderStyle="heavy" borderColor={theme().accent} opacity={0.7} flexShrink={0}>
            <text fg={theme().accent} attributes={TextAttributes.BOLD}>
              {`› ${draft() || "… speaking …"} ●live`}
            </text>
          </box>
        </Show>
        {/* Realtime LLM stream: the assistant's tokens as they arrive,
            rendered as a streaming markdown bubble (final `chat` event
            replaces this preview with the complete message). */}
        <Show when={streamText()}>
          <box flexDirection="column" flexShrink={0} opacity={0.9}>
            <markdown content={streamText().slice(-1200) + "▌"} syntaxStyle={mdStyle} streaming={true} fg={theme().ink} />
          </box>
        </Show>
        {/* The menu belongs to the prompt, so it renders directly above the
            input box — after the flexGrow spacer, not next to the transcript.
            An empty transcript is the common case, and placing it after the
            history left the menu stranded at the top of the panel. */}
        <Show when={cmdOpen()}>
          <box
            flexDirection="column"
            flexShrink={0}
            border
            borderStyle="heavy"
            borderColor={theme().inputBorder}
            backgroundColor={theme().panel}
            marginBottom={1}
          >
            <For each={cmdVisible()}>
              {(c, i) => (
                <box backgroundColor={i() === cmdIndex() ? theme().screen : undefined}>
                  <text
                    fg={i() === cmdIndex() ? theme().accent : theme().dim}
                    attributes={i() === cmdIndex() ? TextAttributes.BOLD : TextAttributes.NONE}
                  >
                    {`${i() === cmdIndex() ? "▸ " : "  "}${c.name}  ${c.hint}`}
                  </text>
                </box>
              )}
            </For>
          </box>
        </Show>
        {/* Status line: one row for every former sys log (mode/mic/cwd/
            warm/confirm acks). Never scrolls, never piles up. The ›› marker
            sets it apart from › user and › think transcript rows. */}
        <Show when={status()}>
          <text fg={theme().dim} attributes={TextAttributes.DIM}>
            {`›› ${status()}`}
          </text>
        </Show>
        <box
          border
          borderStyle="heavy"
          borderColor={theme().inputBorder}
          flexDirection="row"
          flexShrink={0}
          height={inputLines() > 1 ? inputLines() + 2 : undefined}
        >
          <text fg={theme().ink}>{" "}</text>
          <text fg={mode() === "voice" ? theme().voiceBlue : theme().ink}>{mode() === "voice" ? "voice" : mode()}</text>
          <text fg={theme().ink}>{" -> "}</text>
          <input
            flexGrow={1}
            focused
            value={value()}
            onInput={(v: string) => {
              const clean = v.replace(/\t/g, "");
              setValue(clean);
              // Re-arm the palette when the text stops being a command, and
              // reset the highlight as the prefix narrows so Tab completes the
              // top match rather than a stale highlight from a longer prefix.
              if (!clean.startsWith("/") || /\s/.test(clean)) setCmdDismissed(false);
              setCmdSel(0);
            }}
            onSubmit={submitFromInput}
            placeholder={mode() === "voice" ? "live — just speak, no typing needed" : "Add a follow-up  ( / for commands · v for voice )"}
          />
        </box>
        {/* State of execution on right corner below input + permission prompt on left */}
        <box flexDirection="row" justifyContent="space-between" flexShrink={0} marginTop={0}>
          <box>
            <Show when={pending()}>
              {(p: () => Record<string, unknown>) => (
                <text fg={theme().warn} attributes={TextAttributes.BOLD}>
                  {`⬡ Permission: Confirm ${String(p()["action"] ?? "?")}: ${String(p()["command"] ?? p()["path"] ?? "")} (y/n)`}
                </text>
              )}
            </Show>
          </box>
          <box flexDirection="row" justifyContent="flex-end">
            <Show when={phase() || busy()}>
              <text fg={phase() === "interrupted" ? theme().danger : theme().accent} attributes={TextAttributes.BOLD}>
                {`${spinnerFrame(tick())} ${phaseLabel(phase()) || "Thinking"}`}
              </text>
            </Show>
            <Show when={!phase() && !busy() && histOff() > 0}>
              <text fg={theme().dim}>{"↑ scrolled back — PgDn / wheel ↓ for live"}</text>
            </Show>
          </box>
        </box>
      </box>
    </box>
  );
}
