import type { ComponentType } from "react";

export declare const VoiceModeView: ComponentType<{
  micStream: MediaStream | null;
  sid: string;
  apiBase: string;
  onTranscript: (who: "user" | "agent", text: string) => void;
  onError: (msg: string) => void;
}>;
