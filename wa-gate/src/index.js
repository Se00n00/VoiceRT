/* wa-gate: headless WhatsApp gateway for the VoiceAgent contact channel.
 *
 * Baileys keeps one WhatsApp session (sessions/ is gitignored). First boot
 * with no session emits a QR; scan it ONCE from the operator phone
 * (WhatsApp -> Linked Devices -> Link a device). After that the session
 * persists and auto-reconnects; QR is never needed again.
 *
 * HTTP API (binds 127.0.0.1 only):
 *   GET  /wa/health                    {ok, linked, phone, uptime_s, qr_waiting}
 *   GET  /wa/qr                        {linked} | {linked:false, qr_data_url}
 *   POST /wa/send        {to, text}    {ok, err}
 *   POST /wa/send-voice  {to, ogg_b64} {ok, err}   (sent as ptt voice note)
 *   POST /wa/logout                    clear session, back to QR mode
 *
 * Inbound WhatsApp messages are forwarded to WA_INBOUND_URL
 * (default http://127.0.0.1:8003/contact/inbound) as {phone, text}.
 *
 * Env: WA_GATE_PORT (8100), WA_INBOUND_URL, WA_LOG_LEVEL (info).
 */
"use strict";

const express = require("express");
const fs = require("fs");
const os = require("os");
const path = require("path");
const pino = require("pino");
const QRCode = require("qrcode");
const {
  default: makeWASocket,
  DisconnectReason,
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
} = require("@whiskeysockets/baileys");

const PORT = parseInt(process.env.WA_GATE_PORT || "8100", 10);
const INBOUND_URL =
  process.env.WA_INBOUND_URL || "http://127.0.0.1:8003/contact/inbound";
const SESSION_DIR = path.join(__dirname, "..", "sessions", "wa");
const T0 = Date.now();

const log = pino({ level: process.env.WA_LOG_LEVEL || "info" });

let sock = null;
let linked = false;
let linkedPhone = "";
let qrString = "";
let qrAt = 0;
let retries = 0;

function normalizeTo(to) {
  let s = String(to || process.env.WA_DEFAULT_TO || "").trim();
  if (s.includes("@")) return s; // already a JID
  const digits = s.replace(/\D/g, "");
  if (digits.length === 10) return "91" + digits + "@s.whatsapp.net";
  return digits + "@s.whatsapp.net";
}

function jidToPhone(jid) {
  return String(jid || "").split("@")[0].replace(/\D/g, "");
}

function msgText(m) {
  const msg = m.message || {};
  return (
    msg.conversation ||
    (msg.extendedTextMessage && msg.extendedTextMessage.text) ||
    (msg.imageMessage && msg.imageMessage.caption) ||
    (msg.videoMessage && msg.videoMessage.caption) ||
    ""
  ).toString();
}

async function forwardInbound(phone, text) {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 10000);
    const r = await fetch(INBOUND_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ phone, text }),
      signal: ctl.signal,
    });
    clearTimeout(t);
    log.info({ phone, text: text.slice(0, 80), status: r.status }, "inbound fwd");
  } catch (err) {
    log.warn({ err: String(err).slice(0, 200) }, "inbound forward failed");
  }
}

async function connect() {
  fs.mkdirSync(SESSION_DIR, { recursive: true });
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  let version;
  try {
    const v = await fetchLatestBaileysVersion();
    version = v.version;
  } catch {
    version = [2, 3000, 1015901307];
  }
  sock = makeWASocket({
    version,
    auth: state,
    logger: pino({ level: process.env.WA_LOG_LEVEL || "warn" }),
    browser: ["VoiceAgent", "wa-gate", "1.0"],
    printQRInTerminal: false,
  });
  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (u) => {
    const { connection, lastDisconnect, qr } = u;
    if (qr) {
      qrString = qr;
      qrAt = Date.now();
      log.info("QR ready — scan from the operator phone");
    }
    if (connection === "open") {
      linked = true;
      retries = 0;
      qrString = "";
      linkedPhone = jidToPhone(sock.user && sock.user.id);
      log.info({ phone: linkedPhone }, "WhatsApp linked");
    }
    if (connection === "close") {
      linked = false;
      linkedPhone = "";
      const code =
        lastDisconnect &&
        lastDisconnect.error &&
        lastDisconnect.error.output &&
        lastDisconnect.error.output.statusCode;
      log.warn({ code }, "connection closed");
      if (code === DisconnectReason.loggedOut) {
        // session revoked on the phone: wipe and restart to QR mode
        try {
          fs.rmSync(SESSION_DIR, { recursive: true, force: true });
        } catch {}
        retries = 0;
        setTimeout(connect, 2000);
      } else {
        retries += 1;
        setTimeout(connect, Math.min(5000 * retries, 30000));
      }
    }
  });

  const seen = new Set();
  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;
    for (const m of messages || []) {
      if (!m.message || m.key.fromMe) continue;
      const id = m.key.id;
      if (id && seen.has(id)) continue;
      if (id) {
        seen.add(id);
        if (seen.size > 500) {
          const first = seen.values().next().value;
          seen.delete(first);
        }
      }
      const phone = jidToPhone(m.key.remoteJid);
      const text = msgText(m).slice(0, 2000);
      if (!text) continue;
      await forwardInbound(phone, text);
    }
  });
}

// ---- HTTP ---------------------------------------------------------------

const app = express();
app.use(express.json({ limit: "12mb" }));

app.get("/wa/health", (_req, res) => {
  res.json({
    ok: true,
    linked,
    phone: linkedPhone,
    uptime_s: Math.round((Date.now() - T0) / 1000),
    qr_waiting: !linked && !qrString,
  });
});

app.get("/wa/qr", async (_req, res) => {
  if (linked) return res.json({ linked: true, phone: linkedPhone });
  if (!qrString) return res.json({ linked: false, qr: null, waiting: true });
  try {
    const dataUrl = await QRCode.toDataURL(qrString, { width: 320 });
    res.json({
      linked: false,
      qr_data_url: dataUrl,
      age_s: Math.round((Date.now() - qrAt) / 1000),
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err).slice(0, 200) });
  }
});

app.post("/wa/send", async (req, res) => {
  if (!sock || !linked)
    return res.json({ ok: false, err: "not linked: scan QR first" });
  const text = String((req.body || {}).text || "").slice(0, 4000).trim();
  if (!text) return res.json({ ok: false, err: "empty text" });
  try {
    await sock.sendMessage(normalizeTo(req.body.to), { text });
    res.json({ ok: true });
  } catch (err) {
    res.json({ ok: false, err: String(err).slice(0, 300) });
  }
});

app.post("/wa/send-voice", async (req, res) => {
  if (!sock || !linked)
    return res.json({ ok: false, err: "not linked: scan QR first" });
  const b64 = String((req.body || {}).ogg_b64 || "");
  if (!b64 || b64.length > 7_000_000)
    return res.json({ ok: false, err: "missing or oversize ogg_b64" });
  const tmp = path.join(
    os.tmpdir(),
    `wa-voice-${Date.now()}-${Math.random().toString(36).slice(2)}.ogg`
  );
  try {
    fs.writeFileSync(tmp, Buffer.from(b64, "base64"));
    await sock.sendMessage(normalizeTo(req.body.to), {
      audio: { url: tmp },
      mimetype: "audio/ogg; codecs=opus",
      ptt: true,
    });
    res.json({ ok: true });
  } catch (err) {
    res.json({ ok: false, err: String(err).slice(0, 300) });
  } finally {
    try {
      fs.unlinkSync(tmp);
    } catch {}
  }
});

app.post("/wa/logout", async (_req, res) => {
  try {
    if (sock) await sock.logout().catch(() => {});
  } finally {
    try {
      fs.rmSync(SESSION_DIR, { recursive: true, force: true });
    } catch {}
    linked = false;
    linkedPhone = "";
    qrString = "";
    setTimeout(connect, 1000);
    res.json({ ok: true });
  }
});

app.listen(PORT, "127.0.0.1", () => {
  log.info({ port: PORT }, "wa-gate up");
  connect().catch((err) => {
    log.error({ err: String(err).slice(0, 300) }, "initial connect failed");
    setTimeout(connect, 5000);
  });
});
