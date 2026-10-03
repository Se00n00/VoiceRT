# 08 — TUI

The OpenTUI (Solid) client. `voicert`, port 8004 to the bridge.

Version **0.0.1**.

## 8.1 Layout

```
src/tui/
├── package.json          # version, deps, scripts
├── bin/voice-term.js     # the voicert / voice-term shim
├── src/
│   ├── index.tsx         # entry: shade boot note
│   ├── app.tsx           # the App component — commands, keys, layout
│   ├── bridge.ts         # WS + HTTP client for :8004
│   ├── session.ts        # ids + offline fallback titles
│   ├── audio.ts          # mic capture / playback
│   ├── epilogue.ts       # exit card + VERSION
│   ├── opencode.ts       # /opencode spawn
│   └── theme.ts          # shades
└── scripts/
    ├── build.mjs         # Solid + Babel → dist/
    └── test-keys.mjs     # 158 headless checks
```

Built with `node scripts/build.mjs`. Bun is not used.

## 8.2 Commands

`COMMANDS` (`src/app.tsx:68-81`) is the single source of truth — the palette
and `/help` are both generated from it:

```
/new       start a fresh session
/cwd       /cwd PATH — set the working directory
/clear     clear the transcript
/model     /model [name] — list or switch the model
/mode      /mode [auto|voice] — show or set the mode
/mic       /mic [on|off] — toggle the microphone
/voice     /voice [sec] — record a voice turn
/shade     /shade [name|none] — monotonic accent hue
/opencode  /opencode [dir] — spawn opencode here
/template  show 40-turn conversation template
/help      show this list
/quit      (or /q) — leave
```

## 8.3 The palette

Typing `/` opens a menu directly above the input box. It is anchored after the
flex spacer, not at the top of the panel — with `flexGrow` above it, an
anchored-at-top menu stranded itself in the upper panel whenever the transcript
was empty.

Eight rows visible, filtered by prefix. `↑`/`↓` wrap and move the highlight;
`Tab` completes with a trailing space; `Esc` closes without eating your text;
typing an argument closes it.

### Enter does not hijack

This is the important rule. Enter runs the highlighted row **only when you have
moved the highlight**. Otherwise it submits the text you typed.

The reason is concrete: with the palette open on `/mode`, the top match is
`/model`, so a hijacking Enter silently ran a different command than the one on
screen:

```
typed:  /mode
enter  →  "could not reach /model (bridge down?)"
```

A palette that runs something other than what is typed is worse than no palette.
An exact match is never second-guessed; `Tab` is the completion key.

## 8.4 Typos

Unknown commands get a suggestion from `editDistance` (`src/app.tsx:82-98`):

```
/modle  →  did you mean /mode
/clea   →  did you mean /clear
```

## 8.5 Shades

`SHADES = ["blue", "yellow", "red", "orange", "green"]` (`src/theme.ts:68`),
monotonic — each step is warmer than the last. `SHADE_HEX` carries a dark and
light value per shade (`theme.ts:73`).

`VOICE_SHADE` selects one at boot; `none`/`off` restores the base palette.
`/shade [name|none]` changes it live.

The shade reaches the exit card as a hex, not a theme object, because by then
the renderer is destroyed and there is nothing left to resolve colors
(`src/tui/src/epilogue.ts:6-8`).

## 8.6 The epilogue

Printed after OpenTUI tears down the alternate screen, so it is plain stdout
rather than a renderable:

```
  █░█ ▄▀▄ ▀ ▄▀▀ ██▀ █▀█ ▀█▀
  ▀▄▀ ▀▄▀ █ ▀▄▄ █▄▄ █▀▄ ░█░

  Session   Friendly greeting
  Continue  voicert -s ses_xWGk1a2b
  Version   v0.0.1
```

`Voice` is in the terminal's default ink, `RT` in the shade. Two rows, seven
glyphs, stored verbatim rather than reassembled — the glyphs are not fixed
width (`i` is one column, the rest are three), and the two rows share their
space columns exactly, which is what keeps them aligned.

`RT_COL = 19` is the column where the shade starts: five glyphs plus their
gaps. Trailing spaces are trimmed per half before coloring, or the right edge
gets a colored gap.

`VERSION` is asserted equal to `package.json` by the test suite, so bumping one
without the other fails loudly.

## 8.7 Testing

```bash
cd src/tui && npm run typecheck && npm test     # 158 checks
```

`scripts/test-keys.mjs` renders the real component through OpenTUI's test
renderer — no TTY, no GPU, no agent backend — and drives it with the mock
keyboard. It deliberately does **not** press `ctrl+o`, which really spawns a
terminal window.

Two hermeticity rules learned the hard way:

1. `App` and the theme must be imported **dynamically, after** the environment
   is set. Static imports at the top evaluate `VOICE_BRIDGE` before the test can
   point it at a dead port, so the "bridge is down" check passed or failed
   against whatever was on :8004.
2. `arecord` is stubbed on `PATH` so the mic monitor starts cleanly.

Frame text extraction strips the 18-column left panel, then the frame art — by
character class, not regex range, since `┃` `─` `█` `▀` all sit above `U+2500`
and an inverted-range test would also drop real message text.

## 8.8 Transcript: boxes, windowing, scrollback

The left panel is info-and-control (mic, equalizer, model, mode·session,
cwd, theme, shade, bridge state); the right panel is the bordered chat.
Every user turn renders inside its own heavy-bordered box in the input
border color at reduced opacity — agent and system lines stay plain text.

The transcript window is measured in terminal *rows*, not messages: a
bordered user turn costs 3 rows, plain lines cost wrapped-text rows. The
window must fit the space left by the panel chrome exactly, because
overflowing a fixed-height column makes rows paint on top of each other.
Live fills newest-first; scrolled fills forward from a top anchor.

Scrolling is turn-anchored: `PgUp`/wheel-up jumps to the previous user
input (which lands on top, reply below), `PgDn`/wheel-down to the next
one, past the last turn returns to live. `histOff` is the message count
from the end to the top line; `/clear` and `/new` reset it. While
scrolled, the phase line reads `↑ scrolled back — PgDn / wheel ↓ for live`.
Wheel needs `useMouse: true` (`src/index.tsx`); with mouse tracking on,
text selection needs Shift held.

`/template` seeds a 40-turn scripted conversation (tool calls, todos,
confirm gate) for visualizing all of the above; nothing is auto-seeded at
boot, so boot state stays clean.

## 8.9 See also

- [06-processes.md](06-processes.md) — the bridge it talks to
- [04-sessions.md](04-sessions.md) — ids and resume
