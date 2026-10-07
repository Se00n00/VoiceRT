import type { ComponentType } from "react";

export declare const VoiceModeView: ComponentType<{
  stream: MediaStream | null;
  partial: string;
  processing: boolean;
  onMicError: (msg: string) => void;
}>;
