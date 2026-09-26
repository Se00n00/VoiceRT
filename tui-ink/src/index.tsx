import React from "react";
import { render } from "ink";
import { App } from "./app.js";
import { applyScreen, detectTheme, lastProbeAnswered, restoreScreen } from "./theme.js";

const seconds = Number(process.argv[2] ?? process.env.VOICE_SECONDS ?? 5) || 5;
// Theme is resolved before Ink owns stdin (OSC 11 probe needs raw stdin),
// then the terminal backdrop is set solid so no default shows through.
const theme = await detectTheme();
await applyScreen(theme);
const forced = (process.env.VOICE_THEME ?? "").trim().toLowerCase();
const themeNote = forced
  ? `theme: ${theme.mode} (VOICE_THEME=${forced})`
  : lastProbeAnswered
    ? `theme: ${theme.mode} (terminal probe)`
    : `theme: ${theme.mode} (probe silent — VOICE_THEME=light|dark to force)`;
process.on("exit", restoreScreen);
process.on("SIGINT", () => restoreScreen());
process.on("SIGTERM", () => restoreScreen());
const instance = render(<App seconds={seconds} theme={theme} themeNote={themeNote} />);
instance.waitUntilExit().then(restoreScreen, restoreScreen);
