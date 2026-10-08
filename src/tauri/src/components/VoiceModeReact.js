// React island for duplex voice calls over WebRTC (app-owned protocol).
// Plain createElement (no JSX) so vite-plugin-solid leaves this file alone;
// it is mounted into an isolated React root by ReactVoiceMode.tsx.
//
// On mount: mic tracks go into an RTCPeerConnection, a "voice" datachannel
// carries transcripts/state, offer/answer runs over POST /talk/offer, and
// the remote reply track plays through a hidden <audio> element. The beam
// visualizes mic + reply mixed. Unmount tears the whole call down.
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

const STATE_LABEL = {
  listening: "Talk to the agent — I'm listening",
  thinking: "Thinking…",
  speaking: "Speaking… (talk over me to interrupt)",
};

export function VoiceModeView(props) {
  const [secs, setSecs] = React.useState(0);
  const [callState, setCallState] = React.useState("listening");
  const [liveText, setLiveText] = React.useState("");
  const [beamStream, setBeamStream] = React.useState(null);
  const audioRef = React.useRef(null);

  // Elapsed clock for the call.
  React.useEffect(() => {
    const t = window.setInterval(() => setSecs((s) => s + 1), 1000);
    return () => window.clearInterval(t);
  }, []);

  // Owns the whole peer connection for the mount lifetime.
  React.useEffect(() => {
    let dead = false;
    let pc = null;
    let dc = null;
    const onTrack = (ev) => {
      try {
        const remote = ev.streams && ev.streams[0];
        const el = audioRef.current;
        if (el && remote) {
          el.srcObject = remote;
          el.play().catch(() => {});
        }
        const tracks = [];
        try {
          props.micStream.getTracks().forEach((t) => tracks.push(t));
        } catch {
          /* ignore */
        }
        if (remote) {
          try {
            remote.getAudioTracks().forEach((t) => tracks.push(t));
          } catch {
            /* ignore */
          }
        }
        if (!dead && tracks.length > 0) {
          try {
            setBeamStream(new MediaStream(tracks));
          } catch {
            /* ignore */
          }
        }
      } catch {
        /* ignore */
      }
    };
    const setup = async () => {
      try {
        pc = new RTCPeerConnection();
        try {
          props.micStream.getTracks().forEach((t) => pc.addTrack(t, props.micStream));
        } catch (e) {
          props.onError(`mic tracks: ${e instanceof Error ? e.message : String(e)}`);
          return;
        }
        dc = pc.createDataChannel("voice");
        dc.onmessage = (ev) => {
          try {
            const msg = JSON.parse(String(ev.data ?? ""));
            if (msg.type === "partial" && typeof msg.text === "string") {
              const t = msg.text.trim().slice(0, 220);
              setLiveText(t);
              props.onPartial(t);
            } else if (msg.type === "state" && typeof msg.state === "string") {
              setCallState(msg.state);
            } else if (msg.type === "transcript") {
              setLiveText("");
              props.onPartial("");
              props.onTranscript(msg.role === "agent" ? "agent" : "user", String(msg.text ?? ""));
            } else if (msg.type === "barge-in") {
              setCallState("listening");
            }
          } catch {
            /* ignore malformed frames */
          }
        };
        dc.onopen = () => {
          if (!dead) setCallState("listening");
        };
        pc.ontrack = onTrack;
        pc.onconnectionstatechange = () => {
          const st = pc.connectionState;
          if ((st === "failed" || st === "closed") && !dead) {
            props.onError(`call ${st} — re-enter voice mode to retry`);
          }
        };
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        if (pc.iceGatheringState !== "complete") {
          await new Promise((resolve) => {
            const to = window.setTimeout(resolve, 5000);
            const check = () => {
              if (pc.iceGatheringState === "complete") {
                window.clearTimeout(to);
                resolve();
              }
            };
            pc.addEventListener("icegatheringstatechange", check);
          });
        }
        if (dead) return;
        const res = await fetch(`${props.apiBase}/talk/offer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sdp: pc.localDescription.sdp, sid: props.sid }),
        });
        if (!res.ok) {
          props.onError(`offer rejected (${res.status}) — is the backend up?`);
          return;
        }
        const data = await res.json();
        await pc.setRemoteDescription({ type: "answer", sdp: data.sdp });
      } catch (e) {
        if (!dead) props.onError(e instanceof Error ? e.message : String(e));
      }
    };
    setup();
    return () => {
      dead = true;
      try {
        dc && dc.close();
      } catch {
        /* ignore */
      }
      try {
        pc && pc.close();
      } catch {
        /* ignore */
      }
    };
  }, []);

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
        STATE_LABEL[callState] ?? STATE_LABEL.listening,
      ),
    ),
    React.createElement(
      VoiceBeam,
      {
        stream: beamStream,
        processing: callState === "thinking",
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
    React.createElement("audio", { ref: audioRef, autoPlay: true, style: { display: "none" } }),
    React.createElement(
      "div",
      { className: "pointer-events-none relative z-10 -mt-6 flex justify-center px-6" },
      React.createElement(
        "p",
        {
          className:
            "max-w-full truncate rounded-full bg-black/55 px-4 py-1.5 text-center font-mono text-[11px] text-sky-300/90 ring-1 ring-white/10 backdrop-blur",
        },
        liveText || `listening… ${fmtClock(secs)}`,
      ),
    ),
  );
}
