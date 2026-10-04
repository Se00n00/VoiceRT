// Deterministic key-handling checks for the migrated App.
// Run: node scripts/test-keys.mjs
//
// Renders the real component through OpenTUI's test renderer (no TTY, no GPU,
// no agent backend) and drives it with the mock keyboard. Deliberately does
// NOT press ctrl+o — that really spawns a terminal window.
import { execSync } from "node:child_process";
import { mkdtempSync, writeFileSync, chmodSync } from "node:fs";
import { readFile } from "node:fs/promises";
const stripAnsi = (s) => s.replace(/\x1b\[[0-9;]*m/g, "");
import { tmpdir } from "node:os";
import path from "node:path";
import { createComponent, render } from "@opentui/solid";
import { createTestRenderer } from "@opentui/core/testing";
import { parseKeypress } from "@opentui/core";

// Stub arecord so the mic monitor starts cleanly and micOn stays true.
// Unreachable bridge: App skips spawning/warming the backend.
const fakeBin = mkdtempSync(path.join(tmpdir(), "opentui-fakebin-"));
writeFileSync(path.join(fakeBin, "arecord"), "#!/bin/sh\nexec cat >/dev/null\n");
chmodSync(path.join(fakeBin, "arecord"), 0o755);
process.env.PATH = `${fakeBin}:${process.env.PATH}`;
process.env.VOICE_BRIDGE = "http://127.0.0.1:59999";

// App and theme are imported dynamically, AFTER the env above. bridge.ts
// snapshots VOICE_BRIDGE into its module-level API at import time, so a static
// import here would capture the default :8004 instead of the dead port below
// and the whole suite would silently depend on whether a real backend happens
// to be listening on 8004.
const { App } = await import("../dist/app.js");
const { DARK, LIGHT } = await import("../dist/theme.js");

let failures = 0;
function check(name, ok, detail = "") {
  if (ok) console.log(`  ok   ${name}`);
  else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

const setup = await createTestRenderer({ width: 100, height: 30, kittyKeyboard: true });
await render(() => createComponent(App, { seconds: 5, theme: DARK, themeNote: "test" }), setup.renderer);

async function settle(ms = 120) {
  await setup.renderOnce();
  await new Promise((r) => setTimeout(r, ms));
  await setup.flush();
}
const frame = () => setup.captureCharFrame();
const flat = () => frame().replace(/\s+/g, " ");
const badge = () => (flat().match(/\[.\] MIC (ON|OFF)/) ?? ["<no badge>"])[0];
const prompt = () => (flat().match(/(?:auto|voice) ->[^│]*/) ?? ["<no prompt>"])[0].trim();
async function clearInput() {
  // Backspace until the prompt is empty, not a fixed count. Single-letter
  // presses (shift+M/V in a non-empty prompt) type literally into the input,
  // so by later stages the prompt can hold far more than any assumed length.
  // (Mock pressKey has no PageUp entry and would also type literal text, so
  // scrollback uses real escape sequences instead — see below.)
  for (let i = 0; i < 64; i++) {
    if (!prompt().replace(/^(?:auto|voice) ->/, "").trim()) break;
    await setup.mockInput.pressKey("BACKSPACE");
    if (i % 8 === 7) await settle(40);
  }
  await settle(60);
}
async function runCmd(cmd) {
  await clearInput();
  await setup.mockInput.typeText(cmd, 4);
  await settle(70);
  await setup.mockInput.pressKey("RETURN");
  await settle(280);
}

// Transcript text only. Strip the left panel when shown (22 cols + wall),
// drop the frame art. When hidden the rows start at the chat wall and the
// strip regex below matches nothing, so nothing is stripped.
const BOX_ART = /[─-◿▀-▟]/u;
// A line is transcript text only if readable characters survive stripping the
// frame art and the level bars.
const panelLines = () =>
  frame()
    .split("\n")
    .map((l) => l.replace(/^.{22}(?=┃)/, ""))
    .map((l) => l.replace(BOX_ART, " ").replace(/┃/g, " ").trim())
    // The prompt's own row reads "auto -> <placeholder>"; it is chrome, not a
    // message, and it is always the last row. The ›› status line is chrome
    // too (former sys logs live there now, never in the transcript).
    .filter((l) => !/^(auto|voice) -> /.test(l))
    .filter((l) => !l.includes("››"))
    .filter((l) => /[a-z0-9]/i.test(l))
    .map((l) => l.replace(/\s{2,}.*$/, "").trim());
const lastLine = () => panelLines().at(-1) ?? "";
// The ›› status row above the input box (one row, never scrolling).
const statusText = () =>
  frame()
    .split("\n")
    .filter((l) => l.includes("››"))
    .map((l) => l.slice(l.indexOf("››")))
    .join(" ");

// The bridge badge is set by an async health probe, so a fixed sleep races it:
// the panel legitimately shows "…" until the probe answers. Poll instead of
// guessing a delay.
async function waitForBridgeLine(ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (flat().includes("bridge disabled") || flat().includes("bridge down") || flat().includes("● bridge")) return flat();
    await settle(100);
  }
  return flat();
}
await settle(300);
await waitForBridgeLine();

console.log("\nboot (panel hidden by default)");
check("prompt renders the mode", prompt().startsWith("auto ->"), prompt());
check("text-mode placeholder shows", frame().includes("Add a follow-up"));
check("no left panel while hidden", !flat().includes("info & control"));
check("left panel rows do not overflow the frame", frame().replace(/\n$/, "").split("\n").length === 30, `${frame().replace(/\n$/, "").split("\n").length} rows`);

console.log("\n/panel shows and hides the dashboard");
await runCmd("/panel");
check("panel opens on command", statusText().includes("panel shown"), JSON.stringify(statusText()));
check("mic badge starts ON", badge() === "[●] MIC ON", badge());
check("header strip shows mic status", flat().includes("MIC STATUS : ON"));
check("session line shows", flat().includes("auto · "));
check("bridge state shows", flat().includes("bridge down"), (flat().match(/.{0,4}bridge.{0,10}/) ?? ["<none>"])[0]);
check("agent readiness lives in the left panel", /agent (ready|down|starting|failed)/.test(flat()));
await runCmd("/panel");
check("panel closes on command", statusText().includes("panel hidden") && badge() === "<no badge>");

// Both borders are heavy: square corners, thick rules. Guards against a
// regression to "rounded" on the panel or "single" on the prompt.
{
  const lines = frame().split("\n");
  const panelTop = lines.find((l) => l.includes("┏"));
  const panelBottom = lines.find((l) => l.includes("┗"));
  const inputBox = lines.find((l) => /┃\s*┏━+┓\s*┃/.test(l));
  check("panel border is heavy, not rounded", !!panelTop && !!panelBottom && !frame().includes("╭"), panelTop ?? "<no ┏>");
  check("panel sides are heavy verticals", lines.filter((l) => l.trimStart().startsWith("┃")).length >= 2);
  check("input border is heavy too", !!inputBox, inputBox ?? "<no heavy input box>");
  check("input box has no rounded corners", !frame().includes("┌"));
  // Two ┏ rows exist (row 0 carries both the widget top and the panel
  // top; the prompt frame is nested inside the panel's right wall).
  const frameHeads = frame().split("\n").filter((l) => l.includes("┏"));
  check("exactly two heavy-frame rows (top strip + input)", frameHeads.length === 2, `${frameHeads.length}`);
  check("only the top row starts at column 0", frameHeads.filter((l) => l.startsWith("┏")).length === 1, JSON.stringify(frameHeads.map((l) => l.trimStart().slice(0, 4))));
}

console.log("\ntyping");
await setup.mockInput.typeText("hello", 5);
await settle();
check("typed text reaches the input", prompt().includes("hello"), prompt());

console.log("\ntab cycles modes (empty prompt)");
await clearInput();
await setup.mockInput.pressKey("TAB");
await settle();
check("mode cycled to voice", flat().includes("mode → voice"), flat().slice(-200));
check("prompt shows voice", prompt().startsWith("voice ->"), prompt());
check("voice placeholder swapped in", frame().includes("just speak"));
check("tab did not type a literal tab", !prompt().includes("\t"));
await setup.mockInput.pressKey("TAB");
await settle();
check("second tab returns to auto", flat().includes("mode → auto (text + LLM, no voice)"));
check("text placeholder restored", frame().includes("Add a follow-up"));

console.log("\nletter shortcuts are case-insensitive (empty prompt)");
await setup.mockInput.pressKey("M", { shift: true });
await settle();
check("shift+M toggled the mic (status)", statusText().includes("mic off"), JSON.stringify(statusText()));
check("shift+M never typed an M", !prompt().includes("M"), prompt());
await setup.mockInput.pressKey("M", { shift: true });
await settle();
check("shift+M toggles it back", statusText().includes("mic on"), JSON.stringify(statusText()));

console.log("\nshortcuts never leak into a non-empty prompt");
await setup.mockInput.typeText("busy", 5);
await settle();
await setup.mockInput.pressKey("V", { shift: true });
await setup.mockInput.pressKey("M", { shift: true });
await settle();
check("shift+V typed literally", prompt().includes("busyV"), prompt());
check("shift+M typed literally", prompt().includes("busyVM"), prompt());

console.log("\nv outside voice mode explains itself");
await clearInput();
await setup.mockInput.pressKey("V", { shift: true });
await settle();
check("no stray V in the prompt", !prompt().includes("V"), prompt());
check("v explains the mode requirement", flat().includes("Tab to switch"), flat().slice(-220));

console.log("\nhistory scrollback (real sequences: mock pressKey has no PageUp entry and would type literal text)");
const liveLines = () => panelLines().join("\n");
const pageUp = () => setup.renderer.stdin.emit("data", Buffer.from("\x1B[5~"));
const pageDown = () => setup.renderer.stdin.emit("data", Buffer.from("\x1B[6~"));
// Boot no longer seeds the template (backend is back), so seed on demand for scroll depth.
await runCmd("/template");
const liveView = liveLines();
check("live view shows the newest turn", liveView.includes("40 turns"), JSON.stringify(lastLine()));
// Turn-anchored scrolling: one PgUp lands the previous user input on top.
pageUp();
await settle();
check("pageup moves the visible window", liveLines() !== liveView);
check("pageup anchors on the last user turn", (panelLines()[0] ?? "").includes("Great, session over"), JSON.stringify(panelLines()[0]));
check("pageup did not type into the prompt", !prompt().includes("PAGE"), prompt());
// Walk back turn by turn to the first one (bounded loop, not a hard count,
// so template growth doesn't silently break the trip).
for (let i = 0; i < 30 && !(panelLines()[0] ?? "").includes("organize the files"); i++) {
  pageUp();
  await settle();
}
check("deep scrollback reaches the first turn", (panelLines()[0] ?? "").includes("organize the files"), JSON.stringify(panelLines()[0]));
for (let i = 0; i < 30 && liveLines() !== liveView; i++) {
  pageDown();
  await settle();
}
check("pagedown returns to the live view", liveLines() === liveView, `${liveLines().length} vs ${liveView.length} chars`);
check("pagedown keeps the renderer alive", frame().length > 0);

console.log("\nmouse wheel scrolls the transcript (over the chat panel)");
const liveAgain = liveLines();
await setup.mockMouse.scroll(50, 8, "up");
await settle();
// One notch is +3 messages; mode/mic sys lines sit above the template, so
// assert the window moved (deep exclusion is covered by the PgUp block).
check("wheel up scrolls back", liveLines() !== liveAgain);
check("wheel up shows the scroll hint", flat().includes("scrolled back"), flat().slice(-160));
await setup.mockMouse.scroll(50, 8, "down");
await settle();
check("wheel down returns to live", liveLines() === liveAgain);

console.log("\nescape keeps the prompt focused");
await setup.mockInput.pressKey("ESCAPE");
await settle();
check("prompt still focused after escape", prompt().startsWith("auto ->"), prompt());
await setup.mockInput.typeText("still typing", 5);
await settle();
check("typing still works after escape", prompt().includes("still typing"), prompt());

console.log("\ncommand palette (/)");
// Read rows off the raw frame, never off flat(): flat() collapses newlines,
// so a `[^┃]*` scan runs straight past the input box into the border art.
// Menu rows sit inside the palette's own wall, hence the two strips.
const MENU_CMDS = "(?:new|cwd|clear|model|mode|mic|voice|shade|opencode|template|help|quit)";
const menuRows = () =>
  frame()
    .split("\n")
    .map((l) => l.replace(/^.{22}(?=┃)/, "").replace(/^┃ ?/, "").replace(/^┃ ?/, ""))
    .filter((l) => new RegExp(`^[▸ ]\\s*/${MENU_CMDS}\\b`).test(l))
    .map((l) => l.replace(/┃.*$/, "").trimEnd());
const inputText = () => inputRaw().trim();
const inputRaw = () => {
  const row = frame().split("\n").find((l) => /┃ (?:auto|voice) ->/.test(l)) ?? "";
  return (row.match(/(?:auto|voice) -> ([^┃]*)/) ?? ["", ""])[1];
};

await clearInput();
await setup.mockInput.typeText("/", 5);
await settle();
check("typing / opens the command menu", menuRows().length > 0, `${menuRows().length} rows`);
check("the first command is highlighted", menuRows()[0]?.startsWith("▸ /new"), menuRows()[0] ?? "<none>");
check("hints are not clipped to the left-panel width", (menuRows()[1] ?? "").includes("/cwd PATH — set the working directory"), menuRows()[1] ?? "<none>");
check("the menu is capped at 8 rows", menuRows().length === 8, `${menuRows().length}`);
// The list is longer than the 8-row window, so /quit is deliberately not on
// screen until you type it. Assert the cap, not that every name is visible.
check("the rows past the cap are still reachable", menuRows().every((r) => /\/[a-z]+/.test(r)));

console.log("\npalette narrows as you type");
await clearInput();
await setup.mockInput.typeText("/sh", 5);
await settle();
check("a unique prefix leaves one row", menuRows().length === 1, JSON.stringify(menuRows()));
check("that row is /shade", menuRows()[0]?.includes("/shade"), menuRows()[0] ?? "<none>");

await clearInput();
await setup.mockInput.typeText("/mo", 5);
await settle();
check("a shared prefix keeps both matches", menuRows().length === 2, JSON.stringify(menuRows()));

console.log("\narrows move the highlight");
await setup.mockInput.pressKey("ARROW_DOWN");
await settle();
check("down moves to the second row", menuRows()[1]?.startsWith("▸"), menuRows()[1] ?? "<none>");
// Only two rows here, so one more down steps past the end and wraps to row 0.
await setup.mockInput.pressKey("ARROW_DOWN");
await settle();
check("the highlight wraps back to the top", menuRows()[0]?.startsWith("▸"), menuRows()[0] ?? "<none>");
await setup.mockInput.pressKey("ARROW_UP");
await settle();
check("up wraps around to the last row", menuRows()[1]?.startsWith("▸"), menuRows()[1] ?? "<none>");

console.log("\ntab completes the highlighted command");
await clearInput();
await setup.mockInput.typeText("/mo", 5);
await settle();
// Move to /model so the completion is not just the top row by default.
await setup.mockInput.pressKey("ARROW_DOWN");
await settle();
const modeBeforeTab = (flat().match(/(auto|voice) ->/) ?? [""])[0];
await setup.mockInput.pressKey("TAB");
await settle();
// /model and /mode both match "/mo"; one down moves off the top row onto /mode.
check("tab filled in the highlighted name", inputText() === "/mode", JSON.stringify(inputText()));
check("completing closes the menu", menuRows().length === 0, `${menuRows().length}`);
check("completing left a trailing space for the argument", inputRaw().trimEnd() === "/mode" && inputRaw().endsWith(" "), JSON.stringify(inputRaw()));
// flat() still holds "mode →" from the earlier mode-cycling block, so compare
// the live prompt instead of searching the whole transcript.
check("tab did not cycle the mode instead", (flat().match(/(auto|voice) ->/) ?? [""])[0] === modeBeforeTab, `${modeBeforeTab} vs ${(flat().match(/(auto|voice) ->/) ?? [""])[0]}`);

console.log("\nenter submits the text, tab completes it");
// Enter never hijacks an untouched menu: "/" alone is not a command, so it
// reports as unknown rather than silently running the highlighted /new. Tab
// is the completion key; the arrows opt you into "run the selection".
await clearInput();
await setup.mockInput.typeText("/", 5);
await settle();
await setup.mockInput.pressKey("RETURN");
await settle();
check("enter on a bare / does not run the top row", flat().includes("unknown /"), flat().slice(-200));
check("the menu closed after submitting", menuRows().length === 0);
await clearInput();

console.log("\nescape closes the menu but keeps the text");
await clearInput();
await setup.mockInput.typeText("/", 5);
await settle();
check("menu is open", menuRows().length > 0);
await setup.mockInput.pressKey("ESCAPE");
await settle();
check("escape closed the menu", menuRows().length === 0, `${menuRows().length}`);
check("escape kept the typed slash", inputText() === "/", JSON.stringify(inputText()));

console.log("\na non-matching prefix shows no menu");
await clearInput();
await setup.mockInput.typeText("/nope", 5);
await settle();
check("unknown prefix hides the menu", menuRows().length === 0, `${menuRows().length}`);
check("the text is untouched", inputText() === "/nope", JSON.stringify(inputText()));

console.log("\nan argument closes the palette");
await clearInput();
await setup.mockInput.typeText("/cwd /tmp", 5);
await settle();
check("typing an argument hides the menu", menuRows().length === 0, `${menuRows().length}`);

console.log("\nletter shortcuts stay suppressed while a command is being typed");
await clearInput();
await setup.mockInput.typeText("/", 5);
await settle();
const micAtSlash = badge();
await setup.mockInput.pressKey("M", { shift: true });
await settle();
// Same rule as any non-empty prompt: M is text, not the mic toggle. Without
// this the first letter of /mic and /model would never be typeable.
check("shift+M did not toggle the mic", badge() === micAtSlash, `${micAtSlash} -> ${badge()}`);
check("shift+M typed literally instead", inputText() === "/M", JSON.stringify(inputText()));
await clearInput();

console.log("\n/help is generated from the palette list");
await clearInput();
await setup.mockInput.typeText("/help", 5);
await settle();
await setup.mockInput.pressKey("RETURN");
await settle();
check("/help lists the commands", flat().includes("/new") && flat().includes("/shade"), flat().slice(-320));
check("/help mentions the menu", flat().includes("Type / for the menu"), flat().slice(-320));
await clearInput();

// --- every command actually dispatches --------------------------------
// Each case runs the command and asserts on the NEWEST transcript line. The
// window is capped, so comparing counts or slicing by index is unreliable —
// assert on the last line instead, and let boot() settle first so its async
// messages can't be mistaken for command output.
console.log("\nevery command dispatches");
const micBadgeNow = () => (frame().match(/\[.\] MIC (?:ON|OFF)/) ?? [""])[0];
await settle(900); // drain boot()'s async pushes

const CMD_CASES = [
  // /help prints two lines, so match on the whole window rather than the last.
  ["/help", () => panelLines().some((l) => l.includes("/new /cwd /clear"))],
  ["/mode", () => statusText().includes("mode: auto (Tab toggles auto ↔ voice)")],
  ["/mode voice", () => statusText().includes("mode → voice")],
  ["/mode auto", () => statusText().includes("mode → auto")],
  ["/mode bogus", () => lastLine() === "mode must be auto|voice"],
  ["/cwd", () => /cwd \//.test(statusText())],
  ["/cwd /tmp", () => statusText().includes("cwd /tmp")],
  ["/mic off", () => statusText().includes("mic off")],
  ["/mic on", () => statusText().includes("mic on")],
  ["/mic", () => statusText().includes("mic off")],
  // Guard the fix: the guard used to match only the exact strings, so a bad
  // argument fell through to the unknown handler and said "did you mean /mic?"
  ["/mic bogus", () => lastLine() === "usage: /mic [on|off]"],
  ["/shade", () => statusText().includes("● green")],
  ["/shade blue", () => statusText().includes("shade → blue (monotonic; backdrop unchanged)")],
  ["/shade none", () => statusText().includes("shade → none (default palette)")],
  ["/shade junk", () => lastLine().startsWith("usage: /shade")],
  ["/nope", () => lastLine() === "unknown /nope — /help"],
  // Typo suggestions: prefix matching alone missed these entirely.
  ["/modle", () => lastLine().includes("did you mean /mode")],
  ["/clea", () => lastLine().includes("did you mean /clear")],
  ["/voice", () => statusText().includes("voice lives in voice mode")],
  ["/model", () => lastLine().includes("could not reach /model")],
  ["/template", () => panelLines().some((l) => l.includes("40 turns")) && panelLines().some((l) => l.includes("todos: ✓ List files | ✓ Group by type"))],
];
for (const [cmd, verify] of CMD_CASES) {
  await runCmd(cmd);
  check(`${cmd} dispatches`, verify(), JSON.stringify(lastLine()));
}
// /template seeded 20 user turns, each wrapped in its own heavy box on top
// of the panel + input frames.
check("user turns render in input-style boxes", frame().split("\n").filter((l) => l.includes("┏")).length > 2);

console.log("\nenter runs the text you typed, not the top match");
await runCmd("/mode");
check("a bare /mode ran /mode, not /model", statusText().includes("mode: auto (Tab toggles auto ↔ voice)"), JSON.stringify(statusText()));
await clearInput();
await setup.mockInput.typeText("/mo", 4);
await settle();
await setup.mockInput.pressKey("ARROW_DOWN");
await settle();
// "/mo" is not itself a command, so Enter runs the highlighted row — /mode,
// which is what the arrow selected. Pressing Enter again with an exact command
// typed ("/mode") submits that text instead; neither path runs /model.
await setup.mockInput.pressKey("RETURN");
await settle();
check("enter on a prefix runs the highlighted row", statusText().includes("mode:"), JSON.stringify(statusText()));
check("enter never silently ran the other match", !panelLines().includes("could not reach /model"), JSON.stringify(panelLines().slice(-3)));

console.log("\nstatus line owns the sys logs (transcript keeps conversation only)");
await runCmd("/mode voice");
check("ack shows above the input", statusText().includes("mode → voice"), JSON.stringify(statusText()));
check("ack never enters the transcript", !panelLines().some((l) => l.includes("mode → voice")), JSON.stringify(panelLines().slice(-3)));
await runCmd("/mode auto");
check("mode restored for the tests below", statusText().includes("mode → auto"), JSON.stringify(statusText()));

console.log("\ntemplate covers every bubble");
await runCmd("/template");
// Fresh seed shows the tail (turns 18-20); earlier showcase rows need PgUps.
{
  const f = flat();
  check("deny renders", f.includes("✗ denied"));
  check("error renders", f.includes("TTS voice unavailable"));
  check("timestamped thought renders", /› think \[\d{2}:\d{2}:\d{2}\]/.test(f));
  check("preview note went to status, not history", statusText().includes("every bubble type") && !panelLines().some((l) => l.includes("every bubble type")));
}
for (let i = 0; i < 10 && !flat().includes("[queued #2]"); i++) {
  pageUp();
  await settle();
}
{
  const f = flat();
  check("queued badge renders", f.includes("[queued #2]"));
  check("confirm prompt renders", f.includes("Confirm?"));
  // Turn 18 rides along in this window (17 + 18 fit the budget together).
  check("compaction renders", f.includes("context compacted"));
  check("markdown code renders", f.includes("cat CHANGELOG.md"));
}

console.log("\n/clear and /new change state");
await runCmd("/mode voice");
await runCmd("/clear");
check("/clear wiped the transcript", panelLines().every((l) => !l.startsWith("›")), `${panelLines().length} rows left`);
check("/clear shows the home screen", flat().includes("Shift+drag selects"), flat().slice(-200));
await runCmd("/panel");
const sidBefore = (frame().match(/(?:auto|voice) · (\w{8})/) ?? [])[1];
await runCmd("/new");
check("/new rotated the session id", (frame().match(/(?:auto|voice) · (\w{8})/) ?? [])[1] !== sidBefore, `${sidBefore} -> ${(frame().match(/(?:auto|voice) · (\w{8})/) ?? [])[1]}`);
await runCmd("/panel");
check("panel hides again", badge() === "<no badge>");

console.log("\nmodified-key normalisation (why ctrl+o needs ctrlName)");
const kittyCtrlO = parseKeypress("\u001b[15;5u", { useKittyKeyboard: true });
check("kitty reports Ctrl+O as a raw control codepoint", kittyCtrlO?.name === "\u000f", `got ${JSON.stringify(kittyCtrlO?.name)}`);
check("kitty still sets ctrl", kittyCtrlO?.ctrl === true);
const legacyCtrlO = parseKeypress("\u000f");
check("legacy reports Ctrl+O as the letter", legacyCtrlO?.name === "o" && legacyCtrlO?.ctrl === true, `got ${JSON.stringify(legacyCtrlO?.name)} ctrl=${legacyCtrlO?.ctrl}`);
const kittyShiftY = parseKeypress("\u001b[89;2u", { useKittyKeyboard: true });
check("kitty keeps uppercase Y uppercased", kittyShiftY?.name === "Y", `got ${JSON.stringify(kittyShiftY?.name)}`);
check("lowercasing yields the shortcut we match", kittyShiftY?.name.toLowerCase() === "y");
const legacyShiftY = parseKeypress("Y");
check("legacy lowercases Y itself", legacyShiftY?.name === "y", `got ${JSON.stringify(legacyShiftY?.name)}`);

console.log("\nmonotonic shades");
{
  const { SHADES, SHADE_HEX, isShade, withShade, shadeHex, themeNote } = await import("../dist/theme.js");
  check("exactly the five requested shades", SHADES.join(",") === "blue,yellow,red,orange,green", SHADES.join(","));
  const shaded = withShade(DARK, "blue");
  const hues = new Set([shaded.accent, shaded.danger, shaded.warn, shaded.voiceBlue]);
  check("one hue for every chromatic role", hues.size === 1, [...hues].join(" "));
  check("that hue is the shade's hex", shaded.accent === SHADE_HEX.blue.dark);
  check("background is untouched", shaded.screen === DARK.screen && shaded.panel === DARK.panel);
  check("ink and border stay black/white", shaded.ink === DARK.ink && shaded.border === DARK.border);
  check("dim text is untouched", shaded.dim === DARK.dim);
  check("light mode uses the darkened hue", withShade(LIGHT, "blue").accent === SHADE_HEX.blue.light);
  check("null shade returns the base palette", withShade(DARK, null) === DARK);
  check("every shade has both modes", SHADES.every((s) => /^#[0-9a-f]{6}$/.test(SHADE_HEX[s].dark) && /^#[0-9a-f]{6}$/.test(SHADE_HEX[s].light)));
  check("no two shades share a hex", new Set(SHADES.map((s) => SHADE_HEX[s].dark)).size === SHADES.length);
  check("shades do not collide with ink/screen", SHADES.every((s) => [DARK.screen, DARK.ink].includes(SHADE_HEX[s].dark) === false));
  check("isShade rejects junk", !isShade("purple") && !isShade("none") && isShade("orange"));
  check("shadeHex follows the mode", shadeHex("red", "light") === SHADE_HEX.red.light);
  check("theme note mentions the shade", themeNote({ mode: "dark", source: "terminal", theme: DARK }, "green").includes("green"));
  check("no shade leaves the note alone", !themeNote({ mode: "dark", source: "terminal", theme: DARK }).includes("shade:"));
  // initialShade reads the env, so pin and restore rather than assume.
  const { initialShade } = await import("../dist/theme.js");
  const savedShade = process.env.VOICE_SHADE;
  process.env.VOICE_SHADE = "";
  check("unset defaults to green", initialShade() === "green", String(initialShade()));
  process.env.VOICE_SHADE = "none";
  check("none opts out", initialShade() === null, String(initialShade()));
  process.env.VOICE_SHADE = "off";
  check("off opts out", initialShade() === null, String(initialShade()));
  process.env.VOICE_SHADE = "BLUE";
  check("env is trimmed and lowercased", initialShade() === "blue", String(initialShade()));
  process.env.VOICE_SHADE = "  orange  ";
  check("env is trimmed", initialShade() === "orange", String(initialShade()));
  process.env.VOICE_SHADE = "purple";
  check("junk env opts out rather than guessing", initialShade() === null, String(initialShade()));
  if (savedShade === undefined) delete process.env.VOICE_SHADE;
  else process.env.VOICE_SHADE = savedShade;
}

console.log("\nfarewell card");
{
  const { sessionName, normalizeSessionId, newSessionId } = await import("../dist/session.js");
  const { sessionEpilogue } = await import("../dist/epilogue.js");
  const id = newSessionId();
  check("ids are ses_ + 26 base62", /^ses_[0-9A-Za-z]{26}$/.test(id), id);
  check("bare id gets the prefix", normalizeSessionId("abc123XYZ") === "ses_abc123XYZ");
  check("prefixed id is kept", normalizeSessionId("ses_abc123XYZ") === "ses_abc123XYZ");
  check("path traversal is rejected", normalizeSessionId("../../etc/passwd") === null);
  check("short ids are rejected", normalizeSessionId("ab") === null);
  check("two words from the first turn", sessionName("Friendly greeting for everyone") === "Friendly greeting", sessionName("Friendly greeting for everyone"));
  check("filler words are skipped", sessionName("can you please fix the wifi driver") === "Fix wifi", sessionName("can you please fix the wifi driver"));
  check("second word is not title-cased", sessionName("check DNS records") === "Check DNS", sessionName("check DNS records"));
  check("all-filler falls back", sessionName("the a of") === "New session", sessionName("the a of"));
  check("single word is allowed", sessionName("status") === "Status", sessionName("status"));
    const card = sessionEpilogue("Friendly greeting", id);
  check("card says VoiceRT, not opencode", card.includes("Continue") && card.includes(`voicert -s ${id}`) && !card.includes("opencode -s"));
  check("card carries the session name", card.includes("Friendly greeting"));
  check("card carries the version", /Version\s+v\d+\.\d+\.\d+/.test(stripAnsi(card)), stripAnsi(card).slice(-120));
  {
    // The card's version must not drift from the package that ships it.
    const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
    const { VERSION } = await import("../dist/epilogue.js");
    check("card version matches package.json", VERSION === pkg.version, `card=${VERSION} pkg=${pkg.version}`);
  }
  // The wordmark: "Voice" in plain ink, "RT" in the shade. Strip SGR and read
  // the glyphs back, since the split is invisible in a raw string comparison.
  const strip = (s) => s.replace(/\x1b\[[0-9;]*m/g, "");
  const { SHADES: ART_SHADES, SHADE_HEX: ART_HEX } = await import("../dist/theme.js");
  // The wordmark as it should look, row by row. Spaces are significant:
  // they are the inter-glyph gaps and the blank upper rows of x-height
  // letters, so this is compared with spacing intact, not squeezed.
  const VOICERT_ART = [
    " █░█ ▄▀▄ ▀ ▄▀▀ ██▀ █▀█ ▀█▀",
    " ▀▄▀ ▀▄▀ █ ▀▄▄ █▄▄ █▀▄ ░█░",
  ].join("|");
  for (const s of ART_SHADES) {
    const art = strip(sessionEpilogue("X", id, ART_HEX[s].dark))
      .split("\n")
      .slice(1, 3);
    const plain = art.map((l) => l.replace(/^ {2}/, "").replace(/\s+$/, ""));
    check(`${s}: wordmark reads VoiceRT`, plain.join("|") === VOICERT_ART, plain.join("|"));
    check(`${s}: wordmark is two rows`, art.length === 2, `${art.length} rows`);
    const colored = sessionEpilogue("X", id, ART_HEX[s].dark);
    const rt = colored.slice(colored.indexOf("█▀█"));
    check(`${s}: RT half carries the shade SGR`, rt.includes(`\x1b[38;2;${parseInt(ART_HEX[s].dark.slice(1), 16) >> 16 & 255};`), ART_HEX[s].dark);
    const inkHalf = colored.slice(0, colored.indexOf("\x1b[38;2;"));
    check(`${s}: Voice half stays unshaded`, !/\x1b\[38;2;/.test(inkHalf));
  }
  check("no shade leaves RT plain bold", !sessionEpilogue("X", id).includes("\x1b[38;2;"));
  check("garbage hex does not emit a broken SGR", !sessionEpilogue("X", id, "not-a-color").includes("\x1b[38;2;"));
  // exit() only prints when something was said. Mirror that decision here so
  // the "nothing talked -> print nothing" rule is covered headlessly.
  const cardFor = (title, firstUtterance) =>
    firstUtterance || title ? sessionEpilogue(title || sessionName(firstUtterance), id) : "";
  check("no talk and no title prints nothing", cardFor("", "") === "");
  check("model title wins over the offline guess", cardFor("Wifi setup", "can you fix the wifi driver").includes("Wifi setup"));
  check("offline guess is the fallback", cardFor("", "can you fix the wifi driver").includes("Fix wifi"));
}

await setup.renderer.destroy();
// Targeted cleanup: only the stub arecord from this run's temp dir.
try {
  execSync(`pkill -f ${fakeBin}`, { stdio: "ignore" });
} catch {}

console.log(`\n${failures === 0 ? "all checks passed" : `${failures} check(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
