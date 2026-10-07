import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { EqBars, fmtClock } from "./EqBars.js";
import { WarpOrb } from "./WarpOrb.js";
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
// Ai02 port icons (Tabler equivalents, inline stroke style like the rest).
const P_ALERT = "M10 3.5L2.8 16.5h14.4L10 3.5zM10 8v4.2M10 14.8h.01";

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

type MenuId = "plus" | null;

// Ai02 input ported to Solid: tall card (textarea + dictation mic / voice
// orb / square send row) with the + attachment menu. The blue orb enters
// full voice mode (voice-glow overlay); the ghost mic is inline dictation.
// The props signature is unchanged so app.tsx needs no edits.
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
  onVoiceMode: () => void;
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
  const hasText = () => props.value.trim().length > 0;

  return (
    <div class="w-full">
      <div class="relative">
        <div class="rounded-[20px] bg-black/25 p-1.5 backdrop-blur-lg">
        <div
          onClick={() => taRef?.focus()}
          class="flex min-h-[120px] cursor-text flex-col rounded-2xl bg-black/70 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.65)] ring-1 ring-white/10 backdrop-blur-xl"
        >
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

          <div class="relative max-h-[258px] flex-1 overflow-y-auto">
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
              class="min-h-[48px] w-full resize-none whitespace-pre-wrap break-words border-0 bg-transparent p-3 text-[16px] text-white/90 shadow-none outline-none placeholder:text-white/30 focus-visible:ring-0"
            />
          </div>

          <div class="flex min-h-[40px] items-center gap-2 p-2">
            <div class="flex items-center gap-1">
              <button
                title="Add attachments"
                aria-label="Add attachments"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleMenu("plus")();
                }}
                class="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 text-white/55 transition hover:bg-white/10 hover:text-white"
              >
                <I d={P_PLUS} cls="h-4 w-4" />
              </button>
            </div>

            <div class="ml-auto flex items-center gap-1.5">
              <Show when={props.recording}>
                <span class="mr-1 font-mono text-[11px] text-white/50">{fmtTime(props.recSecs)}</span>
                <div class="flex h-4 w-32 items-center justify-center gap-0.5 overflow-hidden sm:w-48">
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
                  <>
                    <button
                      title="Dictation"
                      aria-label="Dictation"
                      onClick={(e) => {
                        e.stopPropagation();
                        props.onMicToggle();
                      }}
                      class="flex h-7 w-7 items-center justify-center rounded-full text-white/55 transition hover:bg-white/10 hover:text-white"
                    >
                      <MicIcon />
                    </button>
                    <Show
                      when={hasText()}
                      fallback={
                        <button
                          title="Voice mode"
                          aria-label="Voice mode"
                          onClick={(e) => {
                            e.stopPropagation();
                            props.onVoiceMode();
                          }}
                          class="voice-blob relative flex h-8 w-8 items-center justify-center overflow-hidden text-white transition active:scale-[0.96]"
                        >
                          <WarpOrb />
                          <span class="relative drop-shadow-[0_1px_2px_rgba(0,0,0,0.6)]">
                            <svg viewBox="0 0 20 20" class="h-5 w-5" fill="currentColor" aria-hidden="true">
                              <rect x="3.5" y="8" width="2.4" height="4" rx="1.2" />
                              <rect x="7.3" y="4.5" width="2.4" height="11" rx="1.2" />
                              <rect x="11.1" y="7" width="2.4" height="6" rx="1.2" />
                              <rect x="14.9" y="9.5" width="2.4" height="4" rx="1.2" />
                            </svg>
                          </span>
                        </button>
                      }
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          props.onSubmit();
                        }}
                        disabled={!canSend()}
                        title="Send"
                        aria-label="Send"
                        class="flex h-8 w-8 items-center justify-center rounded-[10px] bg-white text-black transition hover:bg-white/85 active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        <svg viewBox="0 0 20 20" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                          <path d={P_SEND} />
                        </svg>
                      </button>
                    </Show>
                  </>
                }
              >
                <button
                  title="Stop and dictate"
                  aria-label="Stop and dictate"
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onMicToggle();
                  }}
                  class="flex h-7 w-7 items-center justify-center rounded-full bg-red-500/90 text-white transition hover:bg-red-500"
                >
                  <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor">
                    <rect x="5" y="5" width="10" height="10" rx="2" />
                  </svg>
                </button>
              </Show>
            </div>
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
  // When a real conversation exists, the panel flattens the dither behind
  // it exactly like the mouse hover does.
  const hasChat = () => props.msgs.some((m) => m.who === "you" || m.who === "agent" || m.who === "cot");
  return (
    <div class="relative" data-erase={hasChat() ? "" : undefined}>
      <div
        aria-hidden="true"
        class="absolute -inset-4 rounded-[28px] bg-black/20 backdrop-blur-md"
        style={{
          "mask-image":
            "linear-gradient(to right, transparent, black 32px, black calc(100% - 32px), transparent), linear-gradient(to bottom, transparent, black 32px, black calc(100% - 32px), transparent)",
          "-webkit-mask-image":
            "linear-gradient(to right, transparent, black 32px, black calc(100% - 32px), transparent), linear-gradient(to bottom, transparent, black 32px, black calc(100% - 32px), transparent)",
          "mask-composite": "intersect",
          "-webkit-mask-composite": "source-in",
        }}
      />
      <div class="relative p-4 md:p-5">
      <div class="flex flex-col gap-4">
        <For each={props.msgs}>
          {(m) => (
            <Show
              when={m.who === "cot"}
              fallback={
                <Show
                  when={m.who === "you"}
                  fallback={
                    <div
                      class={`px-1 py-0.5 text-sm ${
                        m.who === "agent"
                          ? "text-white/85"
                          : m.who === "err"
                            ? "text-red-200"
                            : "text-white/45"
                      }`}
                    >
                      <div class="mb-0.5 flex items-center justify-between text-[10px] uppercase tracking-widest opacity-60">
                        <span>{m.who === "agent" ? "Agent" : m.who === "err" ? "Error" : "System"}</span>
                        <Show when={m.who === "agent"}>
                          <button class="normal-case tracking-normal underline" onClick={() => props.onSpeak(m.text)}>
                            Speak
                          </button>
                        </Show>
                      </div>
                      {m.who === "agent" ? <RichText text={m.text} /> : <p class="whitespace-pre-wrap break-words">{m.text}</p>}
                    </div>
                  }
                >
                  <div class="bub-you rounded-2xl bg-[#1b1c22]/70 px-4 py-3 text-sm text-white shadow-lg ring-1 ring-white/15 backdrop-blur-xl">
                    <div class="mb-0.5 text-[10px] uppercase tracking-widest opacity-60">
                      <span>You</span>
                    </div>
                    <RichText text={m.text} />
                  </div>
                </Show>
              }
            >
              <ChainOfThoughtThread steps={m.steps ?? []} />
            </Show>
          )}
        </For>
      </div>
      </div>
    </div>
  );
}
