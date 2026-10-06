// Code block (Solid port of the ai-elements code-block API).
// Same composable pieces — CodeBlock > Body + Item + Content —
// styled for the dark transcript. No header tabs: the block shows the
// active language with a tiny language tag + copy button.
// CodeBlockView is the data-driven shortcut; Transcript uses it to
// render ```fenced``` code in agent/user bubbles.
import { Show, createContext, createSignal, useContext } from "solid-js";
import type { JSX } from "solid-js";

export type CodeItem = { language: string; filename?: string; code: string };

const CbCtx = createContext<{
  data: CodeItem[];
  active: () => string;
  setActive: (v: string) => void;
} | null>(null);

export function CodeBlock(props: { data: CodeItem[]; defaultValue?: string; children: JSX.Element }) {
  const [active, setActive] = createSignal(
    props.defaultValue ?? props.data[0]?.language ?? "text",
  );
  return (
    <CbCtx.Provider value={{ data: props.data, active, setActive }}>
      {props.children}
    </CbCtx.Provider>
  );
}

export function CodeBlockBody(props: { children: (item: CodeItem) => JSX.Element }) {
  const ctx = useContext(CbCtx);
  const item = () => ctx?.data.find((d) => d.language === ctx.active()) ?? ctx?.data[0];
  return <Show when={item()}>{(it) => props.children(it())}</Show>;
}

export function CodeBlockItem(props: { value: string; children: JSX.Element }) {
  const ctx = useContext(CbCtx);
  return <Show when={(ctx?.active() ?? "") === props.value}>{props.children}</Show>;
}

export function CodeBlockContent(props: { language?: string; children: string }) {
  return <CodeBlockView language={props.language} code={props.children} />;
}

/** Data-driven no-header block: mini language tag + copy + mono code. */
export function CodeBlockView(props: { language?: string; filename?: string; code: string }) {
  const [copied, setCopied] = createSignal(false);
  let timer: number | undefined;

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(props.code);
      setCopied(true);
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard blocked — nothing copied */
    }
  }

  return (
    <div class="overflow-hidden rounded-xl bg-[#0b0c0f] ring-1 ring-white/10">
      <div class="flex items-center justify-between px-3 py-1.5">
        <span class="font-mono text-[10px] uppercase tracking-widest text-white/35">
          {props.language || "text"}
        </span>
        <button
          onClick={() => void copy()}
          class="rounded-md px-1.5 py-0.5 font-mono text-[10px] text-white/35 transition hover:bg-white/10 hover:text-white/80"
        >
          {copied() ? "copied" : "copy"}
        </button>
      </div>
      <pre class="thin-scroll overflow-x-auto px-3 pb-3 font-mono text-xs leading-relaxed text-white/80">
        <code>{props.code}</code>
      </pre>
    </div>
  );
}
