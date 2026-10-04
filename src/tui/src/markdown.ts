// Markdown rendering for assistant replies.
//
// OpenTUI's <markdown> renderable needs a SyntaxStyle handle —
// a native resource. Create ONE lazily and reuse it for every
// bubble (rebuilding per message would leak the native side).
import { SyntaxStyle } from "@opentui/core";

let shared: SyntaxStyle | null = null;

/** The shared style table for all markdown bubbles. */
export function markdownStyle(): SyntaxStyle {
  if (!shared) shared = SyntaxStyle.create();
  return shared;
}
