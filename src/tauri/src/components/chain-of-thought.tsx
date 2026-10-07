// Chain of thought (Solid port of the ai-elements chain-of-thought API).
// Same composable pieces — ChainOfThought > Header + Content > Step >
// SearchResults/SearchResult, Image — styled for the dark transcript.
// Step data also has a data-driven form (CotStepData) rendered by
// ChainOfThoughtThread, which is what sample threads use.
import { For, Show, createContext, createSignal, useContext } from "solid-js";
import type { JSX } from "solid-js";

export type CotStatus = "complete" | "active" | "pending";

/** Step marker icon: a pure svg component with an optional class override. */
export type CotIcon = (props: { cls?: string }) => JSX.Element;

export type CotStepData =
  | { kind: "search"; label: string; status: CotStatus; results: string[] }
  | { kind: "image"; label: string; status: CotStatus; src: string; alt: string; caption: string }
  | { kind: "text"; label: string; status: CotStatus };

/** Lucide-style stroke icons (the demo uses lucide-react; this app has no icon deps). */
export function SearchIcon(props: { cls?: string }) {
  return (
    <svg viewBox="0 0 24 24" class={props.cls ?? "h-4 w-4"} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

export function ImageIcon(props: { cls?: string }) {
  return (
    <svg viewBox="0 0 24 24" class={props.cls ?? "h-4 w-4"} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </svg>
  );
}

function ThoughtIcon(props: { cls?: string }) {
  return (
    <svg viewBox="0 0 24 24" class={props.cls ?? "h-4 w-4"} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <circle cx="9" cy="10" r="5" />
      <circle cx="17" cy="17" r="2.5" />
    </svg>
  );
}

function CheckIcon(props: { cls?: string }) {
  return (
    <svg viewBox="0 0 20 20" class={props.cls ?? "h-3 w-3"} fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 10.5l4 4 8-9" />
    </svg>
  );
}

const CotCtx = createContext<{ open: () => boolean; toggle: () => void } | null>(null);

export function ChainOfThought(props: { defaultOpen?: boolean; children: JSX.Element }) {
  const [open, setOpen] = createSignal(props.defaultOpen ?? true);
  return (
    <CotCtx.Provider value={{ open, toggle: () => setOpen((v) => !v) }}>
      <div class="overflow-hidden px-1 py-0.5">{props.children}</div>
    </CotCtx.Provider>
  );
}

export function ChainOfThoughtHeader(props: { children?: JSX.Element }) {
  const ctx = useContext(CotCtx);
  const open = () => ctx?.open() ?? true;
  return (
    <button
      onClick={() => ctx?.toggle()}
      class="flex w-full items-center gap-2 px-4 py-2.5 text-left transition hover:bg-white/[0.03]"
    >
      <span class="text-white/40">
        <ThoughtIcon cls="h-3.5 w-3.5" />
      </span>
      <span class="text-[10px] font-medium uppercase tracking-widest text-white/45">
        {props.children ?? "Chain of thought"}
      </span>
      <span class="ml-auto text-white/30">
        <svg
          viewBox="0 0 16 16"
          class={`h-3 w-3 transition-transform ${open() ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        >
          <path d="M4 6l4 4 4-4" />
        </svg>
      </span>
    </button>
  );
}

export function ChainOfThoughtContent(props: { children: JSX.Element }) {
  const ctx = useContext(CotCtx);
  return (
    <Show when={ctx?.open() ?? true}>
      <div class="flex flex-col gap-3 px-4 pb-4 pt-1">{props.children}</div>
    </Show>
  );
}

function Marker(props: { icon?: CotIcon; status: CotStatus }) {
  const ring =
    props.status === "complete"
      ? "border-emerald-400/40 text-emerald-300"
      : props.status === "active"
        ? "animate-pulse border-sky-400/50 text-sky-300"
        : "border-white/15 text-white/30";
  return (
    <span class={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border bg-black/30 ${ring}`}>
      {props.icon ? (
        props.icon({})
      ) : props.status === "complete" ? (
        <CheckIcon />
      ) : props.status === "active" ? (
        <span class="h-1.5 w-1.5 rounded-full bg-current" />
      ) : (
        <span class="h-1.5 w-1.5 rounded-full border border-current" />
      )}
    </span>
  );
}

export function ChainOfThoughtStep(props: {
  icon?: CotIcon;
  label: string;
  status?: CotStatus;
  children?: JSX.Element;
}) {
  const status = () => props.status ?? "complete";
  return (
    <div class="flex gap-2.5">
      <div class="flex flex-col items-center">
        <Marker icon={props.icon} status={status()} />
        <span class="w-px flex-1 bg-white/10" aria-hidden />
      </div>
      <div class="min-w-0 flex-1 pb-0.5">
        <p class="text-[13px] leading-snug text-white/80">{props.label}</p>
        <Show when={props.children}>
          <div class="mt-1.5">{props.children}</div>
        </Show>
      </div>
    </div>
  );
}

export function ChainOfThoughtSearchResults(props: { children: JSX.Element }) {
  return <div class="flex flex-wrap gap-1.5">{props.children}</div>;
}

export function ChainOfThoughtSearchResult(props: { children: JSX.Element }) {
  return (
    <span class="inline-flex max-w-full items-center gap-1.5 truncate rounded-lg bg-[#1e1f22] px-2 py-1 font-mono text-[11px] text-white/60 ring-1 ring-white/10">
      <span class="h-1 w-1 shrink-0 rounded-full bg-emerald-400/70" />
      <span class="truncate">{props.children}</span>
    </span>
  );
}

export function ChainOfThoughtImage(props: { caption?: string; children: JSX.Element }) {
  return (
    <figure class="m-0">
      {props.children}
      <Show when={props.caption}>
        <figcaption class="mt-1 text-[11px] text-white/40">{props.caption}</figcaption>
      </Show>
    </figure>
  );
}

/** Data-driven thread: renders CotStepData[] through the pieces above. */
export function ChainOfThoughtThread(props: { steps: CotStepData[]; defaultOpen?: boolean }) {
  return (
    <ChainOfThought defaultOpen={props.defaultOpen ?? true}>
      <ChainOfThoughtHeader />
      <ChainOfThoughtContent>
        <For each={props.steps}>
          {(s) =>
            s.kind === "search" ? (
              <ChainOfThoughtStep icon={SearchIcon} label={s.label} status={s.status}>
                <ChainOfThoughtSearchResults>
                  <For each={s.results}>
                    {(r) => <ChainOfThoughtSearchResult>{r}</ChainOfThoughtSearchResult>}
                  </For>
                </ChainOfThoughtSearchResults>
              </ChainOfThoughtStep>
            ) : s.kind === "image" ? (
              <ChainOfThoughtStep icon={ImageIcon} label={s.label} status={s.status}>
                <ChainOfThoughtImage caption={s.caption}>
                  <img
                    src={s.src}
                    alt={s.alt}
                    class="aspect-square h-[150px] rounded-md border border-white/10 object-cover"
                  />
                </ChainOfThoughtImage>
              </ChainOfThoughtStep>
            ) : (
              <ChainOfThoughtStep label={s.label} status={s.status} />
            )
          }
        </For>
      </ChainOfThoughtContent>
    </ChainOfThought>
  );
}
