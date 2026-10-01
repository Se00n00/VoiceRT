import { randomBytes } from "node:crypto";

const B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/**
 * Session ids look like opencode's: `ses_` + 26 base62 chars. Uniqueness is
 * 62^26, so collisions are not a practical concern.
 */
export function newSessionId(): string {
  const bytes = randomBytes(26);
  let out = "";
  for (let i = 0; i < 26; i++) out += B62[bytes[i]! % 62];
  return `ses_${out}`;
}

/** Accept a bare id or a full `ses_…` one, reject anything else. */
export function normalizeSessionId(raw: string): string | null {
  const s = raw.trim();
  if (!s) return null;
  const body = s.startsWith("ses_") ? s.slice(4) : s;
  if (!/^[0-9A-Za-z]{6,64}$/.test(body)) return null;
  return `ses_${body}`;
}

// Words that make a poor session title: they carry no topic, only grammar.
const STOP = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do",
  "does", "for", "from", "get", "give", "has", "have", "he", "her", "him", "his",
  "how", "i", "if", "in", "is", "it", "its", "just", "let", "like", "make", "me",
  "my", "no", "not", "of", "on", "one", "or", "our", "out", "please", "she",
  "should", "so", "some", "that", "the", "their", "them", "then", "there",
  "these", "they", "this", "to", "too", "up", "us", "want", "was", "we", "were",
  "what", "when", "where", "which", "who", "why", "will", "with", "would",
  "you", "your",
]);

/**
 * A two-word name for the session, taken from the first thing the user said.
 * Deterministic and offline: no model call, so it works at exit when the
 * backend is already gone. Falls back to a neutral name when the opening
 * message is empty or all filler.
 */
export function sessionName(firstUserText: string): string {
  const words = (firstUserText || "")
    .replace(/[`*_#>|]/g, " ")
    .split(/[^0-9A-Za-z']+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 1 && !STOP.has(w.toLowerCase()));

  if (words.length === 0) return "New session";
  const a = words[0]!;
  const b = words[1] ?? "";
  // Sentence case, matching the farewell card: "Friendly greeting". Only the
  // leading word is touched, so acronyms survive ("Check DNS").
  const head = a[0]!.toUpperCase() + a.slice(1);
  return b ? `${head} ${b}` : head;
}
