// System-aware theme: dark terminal -> black screen + white content,
// light terminal -> white screen + black content.
//
// Detection order: VOICE_THEME=dark|light override first, then an OSC 11
// background query against the real terminal (kitty/ghostty/wezterm all
// answer it), else dark. Detection runs BEFORE Ink takes over stdin
// (see index.tsx) because it briefly flips stdin into raw mode itself.
export type ThemeMode = "dark" | "light";

export type Theme = {
  mode: ThemeMode;
  screen: string; // solid backdrop behind EVERYTHING (no transparency)
  border: string; // panel borders
  panel: string; // equalizer panel fill (= screen, solid)
  ink: string; // primary content text
  dim: string; // secondary text (think/sys/stream)
  accent: string; // user messages
  danger: string; // errors
  warn: string; // confirm prompts
  voiceBlue: string; // voice-mode label
  inputBorder: string; // prompt box border
};

export const DARK: Theme = {
  mode: "dark",
  screen: "#000000",
  border: "#a1a1aa",
  panel: "#000000",
  ink: "#ffffff",
  dim: "#a1a1aa",
  accent: "#67e8f9",
  danger: "#f87171",
  warn: "#fde047",
  voiceBlue: "#93c5fd",
  inputBorder: "#ffffff",
};

export const LIGHT: Theme = {
  mode: "light",
  screen: "#ffffff",
  border: "#4d4d4d", // black 70%
  panel: "#ffffff",
  ink: "#000000", // black 100%
  dim: "#8c8c8c", // black 45%
  accent: "#000000",
  danger: "#000000",
  warn: "#000000",
  voiceBlue: "#000000",
  inputBorder: "#000000", // black 100%
};

/** Relative luminance 0..1 of an sRGB triple. */
export function luminance(r: number, g: number, b: number): number {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** Parse an OSC 11 reply (`rgb:RRRR/GGGG/BBBB` or `rgb:RR/GG/BB`). */
export function parseOsc11(reply: string): [number, number, number] | null {
  const m = reply.match(/rgb:([0-9a-fA-F]+)\/([0-9a-fA-F]+)\/([0-9a-fA-F]+)/);
  if (!m) return null;
  const scale = (h: string) =>
    Math.round((parseInt(h, 16) / (Math.pow(16, h.length) - 1)) * 255);
  return [scale(m[1]), scale(m[2]), scale(m[3])];
}

/** Ask the terminal for its background color. Null on timeout/refusal. */
function queryTerminalBackground(timeoutMs = 300): Promise<[number, number, number] | null> {
  return new Promise((resolve) => {
    const stdin = process.stdin;
    const stdout = process.stdout;
    if (!stdin.isTTY || !stdout.isTTY) {
      resolve(null);
      return;
    }
    let buf = "";
    let done = false;
    const finish = (v: [number, number, number] | null) => {
      if (done) return;
      done = true;
      clearTimeout(to);
      try { stdin.removeListener("data", onData); } catch { /* never */ }
      try { stdin.setRawMode(false); } catch { /* never */ }
      try { stdin.pause(); } catch { /* never */ }
      resolve(v);
    };
    const to = setTimeout(() => finish(null), timeoutMs);
    const onData = (d: Buffer) => {
      buf += d.toString("utf-8");
      if (buf.includes("\x07") || buf.includes("\x1b\\")) {
        finish(parseOsc11(buf));
      }
      if (buf.length > 256) finish(parseOsc11(buf));
    };
    try {
      stdin.setRawMode(true);
      stdin.resume();
      stdin.on("data", onData);
      stdout.write("\x1b]11;?\x1b\\");
    } catch {
      finish(null);
    }
  });
}

/** True when the last detectTheme() probe got a terminal reply. */
export let lastProbeAnswered = false;

export async function detectTheme(): Promise<Theme> {
  const forced = (process.env.VOICE_THEME ?? "").trim().toLowerCase();
  if (forced === "light") return LIGHT;
  if (forced === "dark") return DARK;
  const bg = await queryTerminalBackground().catch(() => null);
  lastProbeAnswered = !!bg;
  if (!bg) return DARK;
  return luminance(bg[0], bg[1], bg[2]) > 0.5 ? LIGHT : DARK;
}

// --- solid screen: paint the terminal backdrop itself -----------------
// Per-Text backgrounds can't cover gaps (spacer, padding, input guts),
// so unpainted cells show the terminal default (your lavender). Setting
// OSC 11 makes EVERY cell the theme color. Original is restored on exit.
let originalBg: [number, number, number] | null = null;
let applied = false;

/** Set the terminal backdrop to the theme screen. Remembers the old one. */
export async function applyScreen(theme: Theme): Promise<void> {
  if (!process.stdout.isTTY) return;
  const current = await queryTerminalBackground(400).catch(() => null);
  if (current) originalBg = current;
  try {
    process.stdout.write(`\x1b]11;${theme.screen}\x07`);
    applied = true;
  } catch { /* never */ }
}

/** Restore the pre-app backdrop. Sync-safe: also used in `exit` handlers. */
export function restoreScreen(): void {
  if (!applied) return;
  applied = false;
  if (!originalBg) return;
  try {
    const [r, g, b] = originalBg;
    const h = (c: number) =>
      Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, "0");
    process.stdout.write(`\x1b]11;#${h(r)}${h(g)}${h(b)}\x07`);
  } catch { /* never */ }
}
