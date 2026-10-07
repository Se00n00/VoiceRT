// Solid port of the jalco/ui Kbd sculpted keycap (MIT, Justin Levine).
// Same keycap look (3px base edge, top gradient, layered shadows) in our
// own palette — no extra dependencies. Compose with KbdCombo for chords.
import type { JSX } from "solid-js";

export function Kbd(props: { children: JSX.Element; size?: "sm" | "md"; cls?: string }) {
  const size = () => (props.size === "sm" ? "min-h-5 min-w-5 px-1 text-[10px]" : "min-h-6 min-w-6 px-1.5 text-[11px]");
  return (
    <kbd
      class={`inline-flex select-none items-center justify-center rounded-lg border border-white/20 border-b-[3px] border-b-white/25 bg-gradient-to-b from-white/15 to-white/5 font-mono font-medium leading-none text-white/85 shadow-[0_2px_0_0_rgba(255,255,255,0.12),0_3px_6px_-2px_rgba(0,0,0,0.4),inset_0_1px_0_0_rgba(255,255,255,0.12)] ${size()} ${props.cls ?? ""}`}
    >
      {props.children}
    </kbd>
  );
}

export function KbdCombo(props: { keys: string[]; size?: "sm" | "md" }) {
  return (
    <span class="inline-flex items-center gap-1">
      {props.keys.map((k) => (
        <Kbd size={props.size}>{k}</Kbd>
      ))}
    </span>
  );
}
