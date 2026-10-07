// Solid bridge for the voice-glow React island: mounts VoiceModeView into
// an isolated React root and re-renders it whenever the stream or the
// processing flag changes. The mic stream is owned by the app — closing it
// is the app's job on voice-mode exit.
import { createEffect, onCleanup, onMount } from "solid-js";
import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { VoiceModeView } from "./VoiceModeReact.js";

export function ReactVoiceMode(props: {
  stream: () => MediaStream | null;
  partial: () => string;
  processing: () => boolean;
  onMicError: (msg: string) => void;
}) {
  let host!: HTMLDivElement;
  let root: Root | undefined;

  function renderReact(): void {
    root?.render(
      createElement(VoiceModeView, {
        stream: props.stream(),
        partial: props.partial(),
        processing: props.processing(),
        onMicError: props.onMicError,
      }),
    );
  }

  onMount(() => {
    root = createRoot(host);
    renderReact();
  });
  createEffect(() => {
    props.stream();
    props.partial();
    props.processing();
    renderReact();
  });
  onCleanup(() => {
    root?.unmount();
    root = undefined;
  });

  return <div ref={host} class="w-full" />;
}
