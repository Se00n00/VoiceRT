import { createCliRenderer } from "@opentui/core";
import { render } from "@opentui/solid";
import { App } from "./app.js";
import { normalizeSessionId } from "./session.js";
import { initialShade, resolveTheme, themeNote } from "./theme.js";

// Flags: [-s <session_id>] [seconds]   (or VOICE_BRIDGE / VOICE_SECONDS env)
const argv = process.argv.slice(2);
let sessionArg: string | null = null;
let secondsArg: string | null = null;
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "-s" || a === "--session") sessionArg = argv[++i] ?? null;
  else if (a?.startsWith("--session=")) sessionArg = a.slice("--session=".length);
  else if (a?.startsWith("-s=")) sessionArg = a.slice(3);
  else if (!a?.startsWith("-")) secondsArg ??= a;
}

const initialSid = sessionArg ? normalizeSessionId(sessionArg) : null;
if (sessionArg && !initialSid) {
  process.stderr.write(`ignoring malformed session id: ${sessionArg}\n`);
}
const seconds = Number(secondsArg ?? process.env.VOICE_SECONDS ?? 5) || 5;

// OpenTUI owns the terminal: alternate screen, raw input, color detection and
// the solid backdrop. Theme is resolved before the tree mounts so the first
// paint is already correct (no fallback flash).
const renderer = await createCliRenderer({
  exitOnCtrlC: false,
  targetFps: 60,
  autoFocus: false,
  // Mouse tracking for wheel scrolling the transcript. Trade-off: with
  // tracking on, text selection needs Shift held in most terminals.
  useMouse: true,
  useKittyKeyboard: {},
});

const resolved = await resolveTheme(renderer);
// The shade never touches the backdrop: this is the *base* screen color, and
// App layers the hue over the palette at render time.
renderer.setBackgroundColor(resolved.theme.screen);
const shade = initialShade();

await render(
  () => (
    <App
      seconds={seconds}
      theme={resolved.theme}
      themeNote={themeNote(resolved, shade)}
      initialSid={initialSid ?? undefined}
    />
  ),
  renderer,
);
