// Execution-state spinner: rotating braille glyphs driven by the
// app's 120ms tick (one full rotation ≈ 1.2s at 10 frames).
export const SPINNER_FRAMES = [
  "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
] as const;

/** Frame for a tick counter — reads the signal in the render path. */
export function spinnerFrame(tick: number): string {
  return SPINNER_FRAMES[Math.floor(tick) % SPINNER_FRAMES.length] ?? "⠋";
}

/** Human label for the phase line under the input. */
export function phaseLabel(phase: string): string {
  switch (phase) {
    case "thinking …":
      return "Thinking";
    case "LLM …":
      return "LLM";
    case "front model …":
      return "Front model";
    case "worker …":
      return "Worker";
    case "toolcall …":
      return "Tool call";
    case "answering …":
      return "Answering";
    case "speaking …":
      return "Speaking";
    case "recording — speak now":
      return "Dictating";
    case "queued …":
      return "Queued";
    case "interrupted":
      return "Interrupted";
    default:
      return phase.trim() ? phase.replace(/ …$/, "") : "";
  }
}
