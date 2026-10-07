// React island for the voice-glow VoiceBeam (MIT, Jakub Antalik).
// Plain createElement (no JSX) so vite-plugin-solid leaves this file alone;
// it is mounted into an isolated React root by ReactVoiceMode.tsx.
// The microphone stream is owned by the app (voice mode): it is passed in
// via props and never closed here. Open mic tracks stop with the island.
import React from "react";
import { VoiceBeam } from "voice-glow";

function fmtClock(total) {
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function VoiceLogo() {
  return React.createElement(
    "svg",
    {
      viewBox: "0 0 32 32",
      className: "mx-auto h-9 w-9 text-white",
      fill: "none",
      stroke: "currentColor",
      strokeWidth: 2.4,
      strokeLinecap: "round",
      "aria-hidden": true,
    },
    React.createElement("circle", { cx: 15, cy: 16, r: 9 }),
    React.createElement("path", { d: "M15 7 L24 25" }),
  );
}

export function VoiceModeView(props) {
  const [secs, setSecs] = React.useState(0);

  // Elapsed clock for the 60s voice window.
  React.useEffect(() => {
    const t = window.setInterval(() => setSecs((s) => s + 1), 1000);
    return () => window.clearInterval(t);
  }, []);

  // Backend-streamed partial transcript (no browser speech fallback).
  const partial = typeof props.partial === "string" ? props.partial : "";

  return React.createElement(
    "div",
    { className: "flex flex-col" },
    React.createElement(
      "div",
      { className: "pb-1 text-center" },
      React.createElement(VoiceLogo, null),
      React.createElement(
        "p",
        { className: "mt-2 text-[15px] font-semibold text-white" },
        "Voice mode",
      ),
      React.createElement(
        "p",
        { className: "mt-0.5 text-xs text-white/50" },
        "Talk to the agent — I'm listening",
      ),
    ),
    React.createElement(
      VoiceBeam,
      {
        stream: props.stream,
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
    ),
    React.createElement(
      "div",
      { className: "pointer-events-none relative z-10 -mt-6 flex justify-center px-6" },
      React.createElement(
        "p",
        {
          className:
            "max-w-full truncate rounded-full bg-black/55 px-4 py-1.5 text-center font-mono text-[11px] text-sky-300/90 ring-1 ring-white/10 backdrop-blur",
        },
        partial || `listening… ${fmtClock(secs)}`,
      ),
    ),
  );
}
