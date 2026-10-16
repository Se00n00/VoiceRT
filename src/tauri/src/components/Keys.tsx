import { For, Show, createSignal, onMount } from "solid-js";
import { deleteKey, getKeys, storeKey } from "../api/client.js";

const LABELS: Record<string, string> = {
  groq: "Groq",
  gemini: "Gemini",
  anthropic: "Anthropic",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  deepseek: "DeepSeek",
  openai_compat: "Custom endpoint",
};

export type ProviderChoice = { name: string; label: string; model: string };

export function CustomProviderPick(props: {
  choices: ProviderChoice[];
  current: string;
  onSelect: (name: string) => void;
}) {
  const [stored, setStored] = createSignal<Record<string, boolean>>({});
  const [selected, setSelected] = createSignal<string | null>((() => {
    try {
      return localStorage.getItem("voicert.customProvider") ?? null;
    } catch {
      return null;
    }
  })());
  const [input, setInput] = createSignal("");
  const [storeName, setStoreName] = createSignal("");
  const [available, setAvailable] = createSignal(true);
  const [working, setWorking] = createSignal(false);
  const [note, setNote] = createSignal("");

  async function load(): Promise<void> {
    try {
      const s = await getKeys();
      setStored(s.providers ?? {});
      setStoreName(s.store ?? "");
      setAvailable(!!s.available);
    } catch {
      setNote("keys unreachable (bridge down?)");
    }
  }

  function rows(): ProviderChoice[] {
    const seen = new Set<string>();
    const out: ProviderChoice[] = [];
    for (const c of props.choices) {
      if (seen.has(c.name)) continue;
      seen.add(c.name);
      out.push(c);
    }
    for (const p of Object.keys(LABELS)) {
      if (seen.has(p)) continue;
      seen.add(p);
      out.push({ name: p, label: LABELS[p]!, model: "" });
    }
    return out;
  }

  function isCurrent(name: string): boolean {
    const r = rows().find((c) => c.name === name);
    return props.current === name || (!!r && !!r.label && props.current === r.label);
  }

  function pick(name: string): void {
    setSelected(name);
    setInput("");
    setNote("");
    try {
      localStorage.setItem("voicert.customProvider", name);
    } catch {
      /* ignore */
    }
    props.onSelect(name);
  }

  async function save(): Promise<void> {
    const p = selected();
    if (!p) return;
    const key = input().trim();
    if (!key) {
      setNote(`${LABELS[p] ?? p}: paste a key first`);
      return;
    }
    setWorking(true);
    setNote("");
    try {
      const r = await storeKey(p, key);
      if (r.kind === "ok") {
        setInput("");
        setNote(`${LABELS[p] ?? p}: stored in the OS keychain`);
      } else {
        setNote(`${LABELS[p] ?? p}: ${r.message ?? "store failed"}`);
      }
    } catch {
      setNote(`${LABELS[p] ?? p}: store failed (bridge down?)`);
    } finally {
      setWorking(false);
      await load();
    }
  }

  async function clear(): Promise<void> {
    const p = selected();
    if (!p) return;
    setWorking(true);
    setNote("");
    try {
      await deleteKey(p);
      setInput("");
      setNote(`${LABELS[p] ?? p}: cleared`);
    } catch {
      setNote(`${LABELS[p] ?? p}: clear failed (bridge down?)`);
    } finally {
      setWorking(false);
      await load();
    }
  }

  onMount(() => void load().then(() => {
    const names = new Set(rows().map((r) => r.name));
    if (selected() === null || !names.has(selected()!)) {
      const hit = rows().find((r) => isCurrent(r.name));
      setSelected(hit ? hit.name : null);
    }
  }));

  const selLabel = () => (selected() ? (LABELS[selected()!] ?? selected()!) : "");

  return (
    <div class="space-y-2">
      <Show when={!available()}>
        <p class="text-[11px] text-amber-200/80">
          No OS keychain{storeName() ? ` (${storeName()})` : ""} — saving is disabled here. Set KEY env vars or the
          repo .env instead.
        </p>
      </Show>
      <div class="flex flex-wrap gap-1.5">
        <For each={rows()}>
          {(r) => (
            <button
              onClick={() => pick(r.name)}
              title={r.model ? `Switch to ${r.label} (${r.model})` : r.label}
              aria-label={`Select ${r.label}`}
              aria-pressed={selected() === r.name}
              class={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 font-mono text-[11px] ring-1 ring-white/15 transition ${
                selected() === r.name || isCurrent(r.name)
                  ? "bg-white/20 text-white"
                  : "dpanel-btn"
              }`}
            >
              <span
                title={stored()[r.name] ? "key stored" : "key missing"}
                class={`h-1.5 w-1.5 shrink-0 rounded-full ${stored()[r.name] ? "bg-emerald-400" : "bg-white/25"}`}
              />
              {r.label}
              <Show when={r.model}>
                <span class="text-[10px] text-white/40">{r.model}</span>
              </Show>
            </button>
          )}
        </For>
      </div>
      <Show when={selected()}>
        <div class="flex items-center gap-2 rounded-xl bg-white/[0.03] p-2 ring-1 ring-white/10">
          <span class="lbl w-24 shrink-0 truncate">{selLabel()} key</span>
          <input
            type="password"
            autocomplete="off"
            spellcheck={false}
            placeholder={stored()[selected()!] ? "•••••• (stored)" : "paste API key…"}
            value={input()}
            onInput={(e) => setInput(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void save();
            }}
            disabled={!available() || working()}
            class="dpanel-input min-w-0 flex-1 rounded-lg px-2.5 py-1.5 font-mono text-[11px] outline-none ring-1 ring-white/15 disabled:opacity-40"
          />
          <button
            onClick={() => void save()}
            disabled={!available() || working()}
            class="dpanel-btn shrink-0 rounded-lg px-2.5 py-1.5 text-xs disabled:opacity-40"
          >
            {working() ? "…" : "Save"}
          </button>
          <Show when={stored()[selected()!]}>
            <button
              onClick={() => void clear()}
              disabled={working()}
              title={`Clear ${selLabel()} key`}
              aria-label={`Clear ${selLabel()} key`}
              class="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-white/55 transition hover:bg-white/10 hover:text-white disabled:opacity-40"
            >
              <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                <path d="M4 4l8 8M12 4l-8 8" />
              </svg>
            </button>
          </Show>
        </div>
      </Show>
      <Show when={note()}>
        <p class="truncate text-[11px] text-white/50" title={note()}>
          {note()}
        </p>
      </Show>
    </div>
  );
}
