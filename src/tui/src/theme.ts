// System-aware theme: dark terminal -> black screen + white content,
// light terminal -> white screen + black content.
//
// OpenTUI owns terminal color detection and the backdrop now (see
// renderer.waitForThemeMode / renderer.setBackgroundColor), so this module
// only maps a resolved mode -> palette and honors the VOICE_THEME override.
import type { CliRenderer } from "@opentui/core";

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

export function themeFor(mode: ThemeMode): Theme {
  return mode === "light" ? LIGHT : DARK;
}

// --- monotonic shades -----------------------------------------------------
//
// A shade collapses every chromatic role in the palette onto ONE hue, so the
// UI is just that hue plus the existing black and white. Deliberately not a
// "recolor the accent" tweak: a monotonic scheme is only monotonic if nothing
// else smuggles a second hue back in. Background (screen/panel) never moves.
//
// Cost, stated plainly: `danger` and `warn` stop being red and yellow, so an
// error and a confirm prompt are told apart by their glyph and label, not by
// color. That is the point of the mode, not an oversight.

export const SHADES = ["blue", "yellow", "red", "orange", "green"] as const;
export type Shade = (typeof SHADES)[number];

// Per-mode values: the dark set is tuned for a black screen, the light set is
// darkened so it survives white without washing out.
export const SHADE_HEX: Record<Shade, { dark: string; light: string }> = {
  blue: { dark: "#60a5fa", light: "#2563eb" },
  yellow: { dark: "#facc15", light: "#a16207" },
  red: { dark: "#f87171", light: "#b91c1c" },
  orange: { dark: "#fb923c", light: "#c2410c" },
  green: { dark: "#4ade80", light: "#15803d" },
};

export function isShade(v: string): v is Shade {
  return (SHADES as readonly string[]).includes(v);
}

/** The shade's hex for a mode — also what the epilogue paints "RT" with. */
export function shadeHex(shade: Shade, mode: ThemeMode): string {
  return SHADE_HEX[shade][mode];
}

/**
 * Collapse `base` onto a single hue. `null` shade means "no shade": the
 * original two-plus-hue palette, untouched.
 */
export function withShade(base: Theme, shade: Shade | null): Theme {
  if (!shade) return base;
  const hue = SHADE_HEX[shade][base.mode];
  return {
    ...base,
    accent: hue,
    danger: hue,
    warn: hue,
    voiceBlue: hue,
    // ink/border/dim/screen/panel are intentionally inherited: the shade is
    // the only color in town, and the backdrop stays put.
  };
}

/** Starting shade: VOICE_SHADE if it names one, else green. `none` opts out. */
export function initialShade(): Shade | null {
  const raw = (process.env.VOICE_SHADE ?? "").trim().toLowerCase();
  // "none"/"off" is the explicit opt-out. Unset is NOT an opt-out: green is
  // the requested default, so VOICE_SHADE="" must still come up green.
  if (raw === "none" || raw === "off") return null;
  if (raw) return isShade(raw) ? raw : null;
  return "green";
}

export type ThemeSource = "forced" | "terminal" | "default";

export type ResolvedTheme = { mode: ThemeMode; source: ThemeSource; theme: Theme };

/**
 * Resolve the active theme. VOICE_THEME=dark|light forces it; otherwise ask
 * OpenTUI for the terminal's color scheme (it queries the terminal itself),
 * falling back to dark when the terminal stays silent.
 */
export async function resolveTheme(renderer: CliRenderer, timeoutMs = 1000): Promise<ResolvedTheme> {
  const forced = (process.env.VOICE_THEME ?? "").trim().toLowerCase();
  if (forced === "light" || forced === "dark") {
    return { mode: forced, source: "forced", theme: themeFor(forced) };
  }
  const detected = (await renderer.waitForThemeMode(timeoutMs)) ?? renderer.themeMode;
  if (detected === "light" || detected === "dark") {
    return { mode: detected, source: "terminal", theme: themeFor(detected) };
  }
  return { mode: "dark", source: "default", theme: DARK };
}

/** Boot-log line: which theme won and why, plus the active shade. */
export function themeNote(resolved: ResolvedTheme, shade: Shade | null = null): string {
  return shade ? `${baseThemeNote(resolved)} · shade: ${shade}` : baseThemeNote(resolved);
}

function baseThemeNote(resolved: ResolvedTheme): string {
  if (resolved.source === "forced") return `theme: ${resolved.mode} (VOICE_THEME=${resolved.mode})`;
  if (resolved.source === "terminal") return `theme: ${resolved.mode} (terminal probe)`;
  return `theme: ${resolved.mode} (probe silent — VOICE_THEME=light|dark to force)`;
}
