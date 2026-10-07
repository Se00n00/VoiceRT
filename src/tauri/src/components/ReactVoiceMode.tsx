// Solid bridge for the voice-glow React island: mounts VoiceModeView into
// an isolated React root and re-renders it whenever `processing` flips.
// Unmounting (voice mode exit) stops the microphone via the lib itself.
import { createEffect, onCleanup, onMount } from "solid-js";
import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { VoiceModeView } from "./VoiceModeReact.js";

export function ReactVoiceMode(props: { processing: () => boolean; onMicError: (msg: string) => void }) {
  let host!: HTMLDivElement;
  let root: Root | undefined;

  function renderReact(processing: boolean): void {
    root?.render(
      createElement(VoiceModeView, {
        processing,
        onMicError: props.onMicError,
      }),
    );
  }

  onMount(() => {
    root = createRoot(host);
    renderReact(props.processing());
  });
  createEffect(() => {
    renderReact(props.processing());
  });
  onCleanup(() => {
    root?.unmount();
    root = undefined;
  });

  return <div ref={host} class="w-full" />;
}
