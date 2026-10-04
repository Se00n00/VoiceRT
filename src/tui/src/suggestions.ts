// Empty-state home screen: rules, suggestions and help, one line each.
// Shown centered under the wordmark while the transcript is empty
// (dismissed by the first turn, restored by /clear). Plain strings only —
// the empty state renders them with no borders, so the two heavy frames
// (panel + input) stay the only frames on screen.
export const SUGGESTIONS: string[] = [
  "Type / for commands · Tab completes · ↑↓ pick · Enter runs",
  "Tab toggles auto ↔ voice · m toggles the mic · v talks · d dictates",
  "Tools confirm before mutating — nothing destructive runs silently",
  "y confirms a tool call · n or Esc denies it",
  "PgUp / PgDn or the wheel scrolls turn by turn",
  "Shift+drag selects text · Ctrl+C copies the selection",
  "/cwd PATH moves the agent · bare /cwd shows where you are",
  "/model lists sidecars · /model <name> switches (auto-compacts)",
  "/template previews every bubble · /clear wipes · /new starts fresh",
  "Voice lives in voice mode — Tab over and just speak",
];
