import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { EqBars, fmtClock } from "./EqBars.js";
import { ChainOfThoughtThread, type CotStepData } from "./chain-of-thought.js";
import { CodeBlockView } from "./code-block.js";

export type Msg = { id: string; who: "you" | "agent" | "sys" | "err" | "cot"; text: string; steps?: CotStepData[] };

export const COMMANDS: { name: string; hint: string }[] = [
  { name: "/new", hint: "start a fresh session" },
  { name: "/cwd", hint: "/cwd PATH — set the working directory" },
  { name: "/clear", hint: "clear the transcript" },
  { name: "/model", hint: "/model [name] — list or switch the model" },
  { name: "/voice", hint: "/voice [sec] — record a voice turn" },
  { name: "/dictate", hint: "/dictate [sec] — voice dictation input" },
  { name: "/mic", hint: "/mic — record 5s" },
  { name: "/sample", hint: "/sample [name] — load a sample thread" },
  { name: "/help", hint: "show this list" },
];

/** Levenshtein distance for typo suggestions. */
export function editDistance(a: string, b: string): number {
  const prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let last = prev[0] ?? 0;
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j] ?? 0;
      prev[j] = Math.min((prev[j] ?? 0) + 1, (prev[j - 1] ?? 0) + 1, last + (a[i - 1] === b[j - 1] ? 0 : 1));
      last = tmp;
    }
  }
  return prev[b.length] ?? 0;
}

export function suggestTypo(word: string): string | null {
  let best: string | null = null;
  let bestD = 3;
  for (const c of COMMANDS) {
    const d = editDistance(word, c.name);
    if (d < bestD) {
      bestD = d;
      best = c.name;
    }
  }
  return best;
}

export type AgentMode = "Agent" | "Assistant";
export type Perf = "High" | "Medium" | "Low";

const NAV = ["Features", "Platforms", "Insights", "Pricing"];

export function Nav(props: { ok: boolean; model: string; settingsOpen: boolean; onLogin: () => void }) {
  return (
    <div class="topnav flex items-center gap-6 px-7 pt-5 text-[13px] text-white/70">
      <span class="logo flex h-6 w-6 items-center justify-center rounded-full border-[1.5px] border-white/80" aria-label="VoiceRT home">
        <span class={`h-1.5 w-1.5 rounded-full ${props.ok ? "bg-emerald-400" : "bg-red-400"}`} title={props.ok ? "bridge up" : "bridge down"} />
      </span>
      <nav class="mx-auto flex items-center gap-7">
        <For each={NAV}>
          {(item) => (
            <a href="#" onClick={(e) => e.preventDefault()} class="transition hover:text-white">
              {item}
            </a>
          )}
        </For>
      </nav>
      <span class="mlabel hidden font-mono text-[11px] text-white/40 md:inline">{props.model}</span>
      <button onClick={props.onLogin} class="transition hover:text-white">
        {props.settingsOpen ? "Close" : "Login"}
      </button>
    </div>
  );
}

export function Greeting() {
  return (
    <div class="greet text-center">
      <svg viewBox="0 0 32 32" class="mx-auto h-9 w-9 text-white" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
        <circle cx="15" cy="16" r="9" />
        <path d="M15 7 L24 25" />
      </svg>
      <h1 class="mt-4 text-[26px] font-bold tracking-tight text-white">Hey! I'm SuperGrok</h1>
      <p class="mt-1 text-[13px] text-white/45">Tell me everything you need</p>
    </div>
  );
}

function I(props: { d: string; cls?: string }) {
  return (
    <svg viewBox="0 0 20 20" class={props.cls ?? "h-4 w-4"} fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d={props.d} />
    </svg>
  );
}

const P_PLUS = "M10 4v12M4 10h12";
const P_PAPERCLIP = "M16 11l-6.2 6.2a3.7 3.7 0 0 1-5.2-5.2L11 5.6a2.5 2.5 0 0 1 3.5 3.5L8.4 15.2a1.2 1.2 0 0 1-1.7-1.7L12 8.2";
const P_LINK = "M8 12l4-4M9.5 5.5l1.8-1.8a3 3 0 0 1 4.2 4.2L13.7 9.7M10.5 14.5l-1.8 1.8a3 3 0 0 1-4.2-4.2l1.8-1.8";
const P_CLIPBOARD = "M7 4h6v3H7zM6 5H5v13h10V5h-1M6 5a1 1 0 0 1 1-1h0";
const P_TEMPLATE = "M4 4h12v12H4zM4 9h12M9 9v7";
const P_CODE = "M7 7L3.5 10 7 13M13 7l3.5 3L13 13";
const P_GLOBE = "M10 3.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM3.5 10h13M10 3.5c-4 4-4 9 0 13M10 3.5c4 4 4 9 0 13";
const P_HISTORY = "M4 5v4h4M4.5 9a6 6 0 1 1-1 4M10 8v4l2.5 2.5";
const P_BULB = "M10 3a5 5 0 0 1 3 9c-.8.7-1 1.5-1 2.5H8c0-1-.2-1.8-1-2.5a5 5 0 0 1 3-9zM9 17.5h2";
const P_WAND = "M10 2.5v4M10 13.5v4M2.5 10h4M13.5 10h4M5 5l2.5 2.5M12.5 12.5L15 15M15 5l-2.5 2.5M7.5 12.5L5 15";
const P_SEND = "M10 17V3M4 9l6-6 6 6";
const P_LAPTOP = "M4 5h12v8H4zM2.5 16.5h15";
const P_USER = "M10 3.5a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM4 16.5a6 6 0 0 1 12 0";
const P_ROBOT = "M5 8h10v7H5zM5 8l-1.5-2M15 8l1.5-2M8.5 11h.01M11.5 11h.01M8.5 14h3";
const P_BOLT = "M11 2.5L4.5 11H9l-1 6.5L14.5 9H10z";
const P_CHECK = "M4 10.5l4 4 8-9";
const P_X = "M4 4l8 8M12 4l-8 8";
const P_STOP = "M5 5h10v10H5z";
// Ai03 port icons (Tabler equivalents, inline stroke style like the rest).
const P_CLOUD = "M6.5 17a3.75 3.75 0 0 1-.55-7.46 5.25 5.25 0 0 1 10.2-1.35A3.6 3.6 0 0 1 15.5 17h-9z";
const P_CHEV = "M5 7.5l5 5 5-5";
const P_CIRCLE = "M10 16.5a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13z";
const P_ARC = "M10 3.5a6.5 6.5 0 1 0 6.5 6.5";

function MicIcon() {
  return (
    <svg viewBox="0 0 20 20" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
      <rect x="7" y="2.5" width="6" height="9" rx="3" />
      <path d="M4.5 9.5a5.5 5.5 0 0 0 11 0M10 15v2.5" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M2.5 10h4l2.5-6 4 12 2.5-6h2" />
    </svg>
  );
}

const TEMPLATES = [
  { label: "Landing page", text: "Build a landing page with a hero, features, and pricing sections." },
  { label: "Debug an error", text: "Debug this error:\n\n[paste the error]" },
  { label: "Explain code", text: "Explain this code:\n\n[paste the code]" },
];

type MenuId = "plus" | "url" | "tpl" | "model" | "agent" | "perf" | null;

// Ai03 input ported to Solid: rounded card (textarea + circular + / Auto
// pill / circular send) with the Local/Cloud · Agent/Assistant ·
// High/Medium/Low dropdown row underneath. Stripped to the Ai03 feature
// set — no context bar, no mic/EQ recording, no URL/template/clipboard
// entries, no bridge-model list (static Local/Cloud). The props signature
// is unchanged so app.tsx needs no edits; stripped props are ignored.
export function AskCard(props: {
  value: string;
  onInput: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  autoComplete: boolean;
  recording: boolean;
  recLevels: number[];
  recSecs: number;
  onMicToggle: () => void;
  onRecCancel: () => void;
  autoMode: boolean;
  onToggleAuto: () => void;
  think: boolean;
  deepSearch: boolean;
  codeMode: boolean;
  showHistory: boolean;
  onToggleThink: () => void;
  onToggleDeep: () => void;
  onToggleCode: () => void;
  onToggleHistory: () => void;
  agentMode: AgentMode;
  onAgentMode: (m: AgentMode) => void;
  perf: Perf;
  onPerf: (p: Perf) => void;
  models: { name: string; label: string }[];
  modelCurrent: string;
  onModelSelect: (name: string) => void;
  onAttach: (f: File) => void;
  onAttachUrl: (url: string) => void;
  onInsertTemplate: (text: string) => void;
  onPasteClipboard: () => void;
  ctx: { tokens: number; ctx: number; pct: number } | null;
  sessionLabel: string;
}) {
  let fileRef!: HTMLInputElement;
  let taRef!: HTMLTextAreaElement;
  const [menu, setMenu] = createSignal<MenuId>(null);
  // Static Ai03 model picker (bridge model list stripped).
  const [selectedModel, setSelectedModel] = createSignal("Local");

  function autoresize(): void {
    if (!taRef) return;
    taRef.style.height = "auto";
    taRef.style.height = `${Math.min(taRef.scrollHeight, Math.floor(window.innerHeight * 0.25))}px`;
  }
  createEffect(() => {
    props.value;
    autoresize();
  });

  function onKey(e: KeyboardEvent): void {
    if (e.key === "Escape") {
      if (menu()) {
        e.preventDefault();
        setMenu(null);
      }
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      setMenu(null);
      props.onSubmit();
    }
  }

  const toggleMenu = (m: Exclude<MenuId, null>) => () => setMenu((v) => (v === m ? null : m));

  function dropItem(icon: string, label: string, onClick: () => void, dash?: boolean) {
    return (
      <button
        onClick={onClick}
        class="flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-xs text-white/75 transition hover:bg-white/10 hover:text-white"
      >
        <span class="opacity-60">
          {dash ? (
            <svg viewBox="0 0 20 20" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-dasharray="2.5 2">
              <path d={icon} />
            </svg>
          ) : (
            <I d={icon} cls="h-4 w-4" />
          )}
        </span>
        {label}
      </button>
    );
  }

  function fmtTime(s: number): string {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  const canSend = () => props.value.trim().length > 0 && !props.busy;

  return (
    <div class="w-full">
      <div class="relative">
        <div class="overflow-hidden rounded-2xl bg-[#1e1f23] shadow-[0_24px_60px_-12px_rgba(0,0,0,0.65)] ring-1 ring-white/10">
          <input
            ref={fileRef}
            type="file"
            multiple
            class="hidden"
            onChange={(e) => {
              for (const f of Array.from(e.currentTarget.files ?? [])) props.onAttach(f);
              e.currentTarget.value = "";
            }}
          />

          <div class="px-3 pb-2 pt-3">
            <textarea
              ref={taRef}
              value={props.value}
              onInput={(e) => {
                props.onInput(e.currentTarget.value);
                autoresize();
              }}
              onKeyDown={onKey}
              placeholder="Ask anything"
              rows={1}
              class="max-h-[25vh] min-h-10 w-full resize-none border-0 bg-transparent p-0 text-[15px] text-white/90 shadow-none outline-none placeholder:text-white/30 focus-visible:ring-0"
            />
          </div>

          <div class="flex items-center justify-between px-2 pb-2">
            <div class="flex items-center gap-1">
              <button
                title="Add attachments"
                aria-label="Add attachments"
                onClick={toggleMenu("plus")}
                class="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 text-white/55 transition hover:bg-white/10 hover:text-white"
              >
                <I d={P_PLUS} cls="h-4 w-4" />
              </button>

              <button
                onClick={props.onToggleAuto}
                title={props.autoMode ? "Auto: confirmations auto-allowed" : "Auto: ask before acting"}
                class={`flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs transition ${
                  props.autoMode
                    ? "border-white/30 bg-white/15 text-white"
                    : "border-white/10 text-white/55 hover:bg-white/5 hover:text-white/80"
                }`}
              >
                <I d={P_WAND} cls="h-3.5 w-3.5" />
                Auto
              </button>
            </div>

            <div class="ml-auto flex items-center gap-1">
              <Show when={props.recording}>
                <span class="mr-1 font-mono text-[11px] text-white/50">{fmtTime(props.recSecs)}</span>
                <div class="flex h-4 w-48 items-center justify-center gap-0.5 overflow-hidden">
                  <For each={props.recLevels}>
                    {(l) => (
                      <div
                        class="w-0.5 shrink-0 rounded-full bg-white/50 transition-all duration-150"
                        style={{ height: `${Math.round(Math.max(8, Math.min(1, l ?? 0)) * 100)}%` }}
                      />
                    )}
                  </For>
                </div>
              </Show>
              <Show
                when={props.recording}
                fallback={
                  <button
                    title="Voice dictation"
                    aria-label="Voice dictation"
                    onClick={props.onMicToggle}
                    class="flex h-7 w-7 items-center justify-center rounded-full text-white/55 transition hover:bg-white/10 hover:text-white"
                  >
                    <MicIcon />
                  </button>
                }
              >
                <button
                  title="Stop and dictate"
                  aria-label="Stop and dictate"
                  onClick={props.onMicToggle}
                  class="flex h-7 w-7 items-center justify-center rounded-full bg-red-500/90 text-white transition hover:bg-red-500"
                >
                  <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor">
                    <rect x="5" y="5" width="10" height="10" rx="2" />
                  </svg>
                </button>
              </Show>
              <button
                onClick={props.onSubmit}
                disabled={!canSend()}
                title="Send"
                aria-label="Send"
                class="flex h-7 w-7 items-center justify-center rounded-full bg-white text-black transition hover:bg-white/85 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d={P_SEND} />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <Show when={menu() === "plus"}>
          <div class="absolute bottom-full left-0 z-40 mb-2 w-56 overflow-hidden rounded-2xl bg-[#22232a] p-1.5 shadow-2xl ring-1 ring-white/15">
            {dropItem(P_PAPERCLIP, "Attach Files", () => {
              setMenu(null);
              fileRef.click();
            })}
            {dropItem(P_CODE, "Code Interpreter", () => {
              props.onToggleCode();
              setMenu(null);
            })}
            {dropItem(P_GLOBE, "Web Search", () => {
              props.onToggleDeep();
              setMenu(null);
            })}
            {dropItem(P_HISTORY, "Chat History", () => {
              props.onToggleHistory();
              setMenu(null);
            })}
          </div>
        </Show>
      </div>

      <div class="relative flex items-center gap-0 pt-2">
        <button
          onClick={toggleMenu("model")}
          class="flex h-6 items-center gap-1.5 rounded-full px-2 text-xs text-white/55 transition hover:bg-white/10 hover:text-white"
        >
          <I d={selectedModel() === "Cloud" ? P_CLOUD : P_LAPTOP} cls="h-3.5 w-3.5" />
          <span>{selectedModel()}</span>
          <I d={P_CHEV} cls="h-3 w-3 opacity-60" />
        </button>
        <Show when={menu() === "model"}>
          <div class="absolute bottom-full left-0 z-40 mb-2 w-56 overflow-hidden rounded-2xl bg-[#22232a] p-1.5 shadow-2xl ring-1 ring-white/15">
            {dropItem(P_LAPTOP, "Local", () => {
              setSelectedModel("Local");
              setMenu(null);
            })}
            {dropItem(P_CLOUD, "Cloud", () => {
              setSelectedModel("Cloud");
              setMenu(null);
            })}
          </div>
        </Show>

        <button
          onClick={toggleMenu("agent")}
          class="flex h-6 items-center gap-1.5 rounded-full px-2 text-xs text-white/55 transition hover:bg-white/10 hover:text-white"
        >
          <I d={props.agentMode === "Agent" ? P_USER : P_ROBOT} cls="h-3.5 w-3.5" />
          <span>{props.agentMode}</span>
          <I d={P_CHEV} cls="h-3 w-3 opacity-60" />
        </button>
        <Show when={menu() === "agent"}>
          <div class="absolute bottom-full left-0 z-40 mb-2 w-56 overflow-hidden rounded-2xl bg-[#22232a] p-1.5 shadow-2xl ring-1 ring-white/15">
            {dropItem(P_USER, "Agent", () => {
              props.onAgentMode("Agent");
              setMenu(null);
            })}
            {dropItem(P_ROBOT, "Assistant", () => {
              props.onAgentMode("Assistant");
              setMenu(null);
            })}
          </div>
        </Show>

        <button
          onClick={toggleMenu("perf")}
          class="flex h-6 items-center gap-1.5 rounded-full px-2 text-xs text-white/55 transition hover:bg-white/10 hover:text-white"
        >
          <I d={P_BOLT} cls="h-3.5 w-3.5" />
          <span>{props.perf}</span>
          <I d={P_CHEV} cls="h-3 w-3 opacity-60" />
        </button>
        <Show when={menu() === "perf"}>
          <div class="absolute bottom-full left-0 z-40 mb-2 w-56 overflow-hidden rounded-2xl bg-[#22232a] p-1.5 shadow-2xl ring-1 ring-white/15">
            {dropItem(P_CIRCLE, "High", () => {
              props.onPerf("High");
              setMenu(null);
            })}
            {dropItem(P_ARC, "Medium", () => {
              props.onPerf("Medium");
              setMenu(null);
            })}
            {dropItem(P_CIRCLE, "Low", () => {
              props.onPerf("Low");
              setMenu(null);
            }, true)}
          </div>
        </Show>
      </div>

      <Show when={menu() !== null}>
        <button aria-hidden tabIndex={-1} onClick={() => setMenu(null)} class="fixed inset-0 z-30 cursor-default bg-transparent" />
      </Show>
    </div>
  );
}

type RichPart = { kind: "text"; text: string } | { kind: "code"; language: string; code: string };

/** Splits ```fenced``` code out of a bubble so generated code renders as blocks. */
function splitRich(text: string): RichPart[] {
  const parts: RichPart[] = [];
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push({ kind: "text", text: text.slice(last, m.index) });
    parts.push({
      kind: "code",
      language: (m[1] || "text").toLowerCase(),
      code: (m[2] ?? "").replace(/\n$/, ""),
    });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts.filter((p) => (p.kind === "text" ? p.text.length > 0 : true));
}

function RichText(props: { text: string }) {
  const parts = splitRich(props.text);
  return (
    <>
      <For each={parts}>
        {(p) =>
          p.kind === "code" ? (
            <div class="mb-1 mt-2 first:mt-0">
              <CodeBlockView language={p.language} code={p.code} />
            </div>
          ) : (
            <p class="whitespace-pre-wrap break-words">{p.text}</p>
          )
        }
      </For>
    </>
  );
}

export function Transcript(props: { msgs: Msg[]; onSpeak: (text: string) => void }) {
  return (
    <div class="flex flex-col gap-2.5">
      <For each={props.msgs}>
        {(m) => (
          <Show
            when={m.who === "cot"}
            fallback={
              <div
                class={`bub rounded-2xl px-4 py-3 text-sm shadow-lg backdrop-blur ${
                  m.who === "you"
                    ? "bub-you bg-[#1b1c22] text-white ring-1 ring-white/15"
                    : m.who === "agent"
                      ? "bub-agent bg-[#f2f3f5] text-slate-700"
                      : m.who === "err"
                        ? "bub-err bg-[#2a0f14] text-red-100 ring-1 ring-red-400/30"
                        : "bub-sys bg-[#101116] text-white/45"
                }`}
              >
                <div class="mb-0.5 flex items-center justify-between text-[10px] uppercase tracking-widest opacity-60">
                  <span>{m.who === "you" ? "You" : m.who === "agent" ? "Agent" : m.who === "err" ? "Error" : "System"}</span>
                  <Show when={m.who === "agent"}>
                    <button class="normal-case tracking-normal underline" onClick={() => props.onSpeak(m.text)}>
                      Speak
                    </button>
                  </Show>
                </div>
                {m.who === "agent" || m.who === "you" ? (
                  <RichText text={m.text} />
                ) : (
                  <p class="whitespace-pre-wrap break-words">{m.text}</p>
                )}
              </div>
            }
          >
            <ChainOfThoughtThread steps={m.steps ?? []} />
          </Show>
        )}
      </For>
    </div>
  );
}
