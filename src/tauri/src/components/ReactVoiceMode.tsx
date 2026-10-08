// Solid bridge for the duplex voice-call React island: mounts
// VoiceModeView into an isolated React root. The island owns the whole
// RTCPeerConnection for its mount lifetime; unmount hangs the call up.
import { createEffect, onCleanup, onMount } from "solid-js";
import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { VoiceModeView } from "./VoiceModeReact.js";

export function ReactVoiceMode(props: {
  micStream: () => MediaStream | null;
  sid: () => string;
  apiBase: string;
  onTranscript: (who: "user" | "agent", text: string) => void;
  onError: (msg: string) => void;
}) {
  let host!: HTMLDivElement;
  let root: Root | undefined;

  function renderReact(): void {
    root?.render(
      createElement(VoiceModeView, {
        micStream: props.micStream(),
        sid: props.sid(),
        apiBase: props.apiBase,
        onTranscript: props.onTranscript,
        onError: props.onError,
      }),
    );
  }

  onMount(() => {
    root = createRoot(host);
    renderReact();
  });
  createEffect(() => {
    props.micStream();
    renderReact();
  });
  onCleanup(() => {
    root?.unmount();
    root = undefined;
  });

  return <div ref={host} class="w-full" />;
}
