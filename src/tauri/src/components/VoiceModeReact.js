// React island for the voice-glow VoiceBeam (MIT, Jakub Antalik).
// Plain createElement (no JSX) so vite-plugin-solid leaves this file alone;
// it is mounted into an isolated React root by ReactVoiceMode.tsx.
// The microphone is owned by the lib itself (useMicrophone): the stream is
// requested on mount and stopped on unmount. Beam-only: entering/exiting
// voice mode is handled by the app's bottom panel button.
import React from "react";
import { VoiceBeam, useMicrophone } from "voice-glow";

export function VoiceModeView(props) {
  const mic = useMicrophone({ autoStart: true });
  const state = mic.state;

  React.useEffect(() => {
    if (state === "denied" || state === "error" || state === "unsupported") {
      props.onMicError(
        state === "denied"
          ? "microphone denied — allow mic access, then re-enter voice mode."
          : state === "unsupported"
            ? "microphone capture is not supported in this browser."
            : "microphone error — re-enter voice mode to retry.",
      );
    }
  }, [state]);

  return React.createElement(
    VoiceBeam,
    {
      stream: mic.stream,
      processing: props.processing,
      type: "mobile",
      theme: "dark",
      colorVariant: "ocean",
      sensitivity: 5.5,
      threshold: 0.01,
      strength: 1,
      className: "pointer-events-none",
    },
    React.createElement("div", { style: { height: 375, width: "100%" } }),
  );
}
