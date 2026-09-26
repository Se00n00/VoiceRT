// Live terminal mirror: runs tui-ink in a pty, streams it to xterm.js.
// Dev-only. Usage: npm install && npm start  (then open http://127.0.0.1:8090)
// Edit tui-ink/src/* -> this server kills + respawns the pty and tells the
// page to clear, so you never see stacked ghost screens. Restart ownership
// lives HERE (plain `tsx`, not `tsx watch`) so every reboot is announced.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pty from "node-pty";
import { WebSocketServer } from "ws";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TUI_DIR = path.resolve(__dirname, "..", "tui-ink");
const SRC_DIR = path.join(TUI_DIR, "src");
const PORT = Number(process.env.PREVIEW_PORT ?? 8090);
// Override for smoke tests, e.g. PREVIEW_CMD="node ticker.mjs".
const CMD = process.env.PREVIEW_CMD
  ? process.env.PREVIEW_CMD.split(" ")
  : ["npm", "run", "dev", "--silent"];

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
};
// UMD builds served straight from node_modules (offline-safe).
const VENDOR = {
  "/vendor/xterm.js": ["@xterm", "xterm", "lib", "xterm.js"],
  "/vendor/xterm.css": ["@xterm", "xterm", "css", "xterm.css"],
  "/vendor/addon-fit.js": ["@xterm", "addon-fit", "lib", "addon-fit.js"],
};

function serveFile(res, file, mime) {
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "Content-Type": mime }).end(data);
  });
}

// --- shared pty: one app, many viewers --------------------------------
// A pty has no real terminal behind it, so OSC 11 never answers and the
// app would always fall back to dark. The page reports the OS scheme
// (?theme= on connect, {t:'theme'} on change) and we inject VOICE_THEME.
let term = null;
let expectExit = false;
let respawnTimer = null;
let ptyTheme = null; // 'light' | null (null = auto, i.e. dark fallback)
const clients = new Set();
const broadcast = (obj) => {
  const msg = JSON.stringify(obj);
  for (const ws of clients) {
    try {
      if (ws.readyState === 1) ws.send(msg);
    } catch { /* dead socket */ }
  }
};

function spawnTerm(cols = 100, rows = 30) {
  if (term) {
    expectExit = true;
    try { term.kill(); } catch { /* already dead */ }
    term = null;
  }
  term = pty.spawn(CMD[0], CMD.slice(1), {
    name: "xterm-256color",
    cols,
    rows,
    cwd: TUI_DIR,
    env: {
      ...process.env,
      ...(ptyTheme ? { VOICE_THEME: ptyTheme } : { VOICE_THEME: "" }),
      TERM: "xterm-256color",
      FORCE_COLOR: "1",
    },
  });
  term.onData((d) => broadcast({ t: "out", d }));
  term.onExit(({ exitCode }) => {
    term = null;
    if (expectExit) {
      expectExit = false;
      return; // intentional kill (file-change respawn) — spawn follows
    }
    // Crash: revive after a breath so the page never sits dead.
    // A fresh `restarted` tells the page to clear first.
    if (respawnTimer) clearTimeout(respawnTimer);
    respawnTimer = setTimeout(() => {
      spawnTerm(cols, rows);
      broadcast({ t: "restarted" });
    }, 1500);
    broadcast({ t: "exit", code: exitCode });
  });
  return term;
}

function respawn(cols = 100, rows = 30) {
  spawnTerm(cols, rows);
  broadcast({ t: "restarted" });
}

// src/ is flat, so a single non-recursive watch covers it (Linux fs.watch
// has no recursive mode).
let watchDebounce = null;
try {
  fs.watch(SRC_DIR, (event, name) => {
    if (!name || !/\.(tsx?|jsx?|css)$/.test(name)) return;
    if (watchDebounce) clearTimeout(watchDebounce);
    watchDebounce = setTimeout(() => respawn(), 400);
  });
} catch (err) {
  console.log(`file watch off (${err.message}); use restart via API`);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  if (url.pathname === "/" || url.pathname === "/index.html") {
    serveFile(res, path.join(__dirname, "public", "index.html"), MIME[".html"]);
    return;
  }
  if (url.pathname === "/api/status") {
    res.writeHead(200, { "Content-Type": "application/json" }).end(
      JSON.stringify({ running: term !== null })
    );
    return;
  }
  const vend = VENDOR[url.pathname];
  if (vend) {
    const ext = path.extname(url.pathname);
    serveFile(res, path.join(__dirname, "node_modules", ...vend), MIME[ext] ?? "application/octet-stream");
    return;
  }
  res.writeHead(404).end("not found");
});

const wss = new WebSocketServer({ server, path: "/pty" });
wss.on("connection", (ws, req) => {
  clients.add(ws);
  // First paint wins: ?theme= from the page seeds the pty env so the app
  // boots in the right mode with no respawn dance.
  try {
    const q = new URL(req?.url ?? "/", "http://x").searchParams.get("theme");
    if ((q === "light" || q === "dark") && !term) ptyTheme = q === "light" ? "light" : null;
  } catch { /* keep current */ }
  if (!term) {
    spawnTerm();
    broadcast({ t: "restarted" });
  } else ws.send(JSON.stringify({ t: "running" }));
  ws.on("message", (raw) => {
    let m;
    try { m = JSON.parse(String(raw)); } catch { return; }
    if (m.t === "in" && term) {
      try { term.write(String(m.d ?? "")); } catch { /* dead pty */ }
    } else if (m.t === "resize" && term) {
      try {
        term.resize(Math.max(20, +m.cols || 100), Math.max(10, +m.rows || 30));
      } catch { /* dead pty */ }
    } else if (m.t === "restart") {
      respawn(Math.max(20, +m.cols || 100), Math.max(10, +m.rows || 30));
    } else if (m.t === "theme") {
      // OS scheme flipped mid-session: re-env + respawn on mismatch.
      const want = m.mode === "light" ? "light" : null;
      if (want !== ptyTheme) {
        ptyTheme = want;
        respawn();
      }
    }
  });
  ws.on("close", () => clients.delete(ws));
});

function shutdown() {
  expectExit = true;
  if (respawnTimer) clearTimeout(respawnTimer);
  try { term?.kill(); } catch { /* never */ }
  wss.close();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000).unref();
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

server.listen(PORT, "127.0.0.1", () => {
  console.log(`terminal mirror at http://127.0.0.1:${PORT}/  (pty: tui-ink, server-owned restarts)`);
});
