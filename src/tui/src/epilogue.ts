// Farewell card printed after OpenTUI tears down the alternate screen.
// Mirrors opencode's `sessionEpilogue` (packages/tui/src/util/presentation.ts)
// but with the VoiceRT wordmark and `voicert -s <id>` as the resume command.
//
// This is plain stdout, not an OpenTUI renderable: the renderer is already
// destroyed by the time we print, so the normal screen buffer is ours again.
// That is also why the shade arrives as a hex -> SGR pair (see ansiFg) rather
// than a theme object: nothing is left to resolve colors for us.

// The wordmark: "Voice" in the terminal's default ink, "RT" in the shade.
//
// Two rows, seven glyphs, single space between glyphs: V o i c e R T. Glyphs
// are NOT fixed width here (`i` is 1 column, the rest are 3), so the rows are
// stored verbatim rather than reassembled from a per-letter table — the two
// rows share their space columns exactly, which is what keeps them aligned.
// ░ (light shade) gives the strokes their bevel.
// Reused by the TUI empty state (center of the chat panel): the same two
// rows render there with "RT" in the live shade color.
export const WORDMARK = [
  " █░█ ▄▀▄ ▀ ▄▀▀ ██▀ █▀█ ▀█▀",
  " ▀▄▀ ▀▄▀ █ ▀▄▄ █▄▄ █▀▄ ░█░",
];
/** Column where "RT" starts: 5 glyphs ("Voice") plus their gaps. */
export const RT_COL = 19;

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[90m";

/**
 * Released version, kept in step with package.json by
 * `check("the card matches package.json", ...)` in scripts/test-keys.mjs —
 * that test reads both files, so bumping one without the other fails loudly
 * rather than shipping a card that lies about which build it came from.
 */
export const VERSION = "0.0.1";

/** "#4ade80" -> a 24-bit foreground SGR. Falls back to the default ink. */
function ansiFg(hex: string): string {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return "";
  const n = parseInt(m[1], 16);
  // eslint-disable-next-line no-bitwise
  return `\x1b[38;2;${(n >> 16) & 255};${(n >> 8) & 255};${n & 255}m`;
}

/**
 * @param title   Two-word session name (see session.ts).
 * @param sessionId  `ses_…` id to resume with.
 * @param shadeHex  Shade hex for the "RT" half; omit for plain bold.
 */
export function sessionEpilogue(title: string, sessionId: string, shadeHex?: string): string {
  const fg = shadeHex ? ansiFg(shadeHex) : "";
  const weak = (text: string) => `${DIM}${text.padEnd(10, " ")}${RESET}`;
  const wordmark: string[] = [];
  for (const row of WORDMARK) {
    // Trailing spaces carry no ink and would leave a colored gap on the
    // right edge, so trim each half independently before wrapping it.
    const inkPart = `${BOLD}${row.slice(0, RT_COL).trimEnd()}${RESET}`;
    const shade = row.slice(RT_COL).trimEnd();
    const shadePart = shade ? `${fg}${BOLD}${shade}${RESET}` : "";
    // Single space: RT_COL lands on a glyph boundary that already had its
    // inter-glyph gap trimmed, so one space restores the original spacing.
    wordmark.push(`  ${inkPart}${shadePart ? ` ${shadePart}` : ""}`);
  }
  return [
    "",
    ...wordmark,
    "",
    `  ${weak("Session")}${BOLD}${title}${RESET}`,
    `  ${weak("Continue")}${BOLD}voicert -s ${sessionId}${RESET}`,
    `  ${weak("Version")}${DIM}v${VERSION}${RESET}`,
    "",
  ].join("\n");
}

export function printEpilogue(title: string, sessionId: string, shadeHex?: string): void {
  process.stdout.write(sessionEpilogue(title, sessionId, shadeHex) + "\n");
}