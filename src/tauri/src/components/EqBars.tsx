import { For } from "solid-js";

/** Shared mic-level equalizer (voice dictation). Heights are 0..1 fractions. */
export function EqBars(props: { levels: number[]; active: boolean; class?: string }) {
  return (
    <div class={`flex items-center gap-[3px] ${props.class ?? "h-4 w-64"}`}>
      <For each={props.levels}>
        {(lv) => (
          <div
            class={`w-0.5 min-w-[2px] flex-1 rounded-full transition-all duration-300 ${
              props.active ? "animate-pulse bg-white/50" : "bg-white/10"
            }`}
            style={{ height: props.active ? `${Math.round(Math.min(1, Math.max(0.06, lv)) * 100)}%` : "4px" }}
          />
        )}
      </For>
    </div>
  );
}

export function fmtClock(s: number): string {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
