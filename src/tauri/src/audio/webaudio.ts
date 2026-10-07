// WebAudio mic capture + TTS playback. Replaces arecord/aplay subprocesses.
import workletUrl from "./recorder-worklet.js?url";

function encodePcm16(chunks: Float32Array[]): Uint8Array {
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const flat = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    flat.set(c, off);
    off += c.length;
  }
  const out = new Uint8Array(flat.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < flat.length; i++) {
    const v = Math.max(-1, Math.min(1, flat[i] ?? 0));
    view.setInt16(i * 2, Math.round(v * 32767), true);
  }
  return out;
}

function audioCtx(sr: number): AudioContext {
  const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  return new Ctx({ sampleRate: sr });
}

export type Recorder = {
  /** Stop capture and resolve the PCM recorded so far. */
  stop: () => Promise<{ pcm: Uint8Array; sr: number }>;
  /** Abort capture and drop everything. */
  cancel: () => void;
};

/** Controllable mic recorder: stop() whenever the user is done (dictation UI).
    Capture runs on an AudioWorklet (ScriptProcessor is deprecated and drops
    audio in some WebViews); falls back to ScriptProcessor if the worklet
    cannot load. */
export async function createRecorder(onLevel?: (rms: number) => void): Promise<Recorder> {
  const sr = 16000;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  const ctx = audioCtx(sr);
  const src = ctx.createMediaStreamSource(stream);
  const chunks: Float32Array[] = [];
  let done = false;
  const feed = (ch: Float32Array) => {
    if (done) return;
    chunks.push(new Float32Array(ch));
    if (onLevel) {
      let sum = 0;
      for (let i = 0; i < ch.length; i += 8) sum += (ch[i] ?? 0) ** 2;
      onLevel(Math.min(1, Math.sqrt(sum / Math.ceil(ch.length / 8)) * 4));
    }
  };
  let teardown = () => {
    stream.getTracks().forEach((t) => t.stop());
  };
  try {
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, "voicert-recorder", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    // Muted tail so the worklet is pulled without ever playing the mic.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (ev: MessageEvent) => feed(ev.data as Float32Array);
    src.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    const prevTeardown = teardown;
    teardown = () => {
      try {
        node.port.onmessage = null;
        src.disconnect();
        node.disconnect();
        mute.disconnect();
      } catch {
        /* ignore */
      }
      prevTeardown();
    };
  } catch {
    // Fallback: legacy ScriptProcessor path.
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    src.connect(proc);
    proc.connect(ctx.destination);
    proc.onaudioprocess = (ev) => feed(ev.inputBuffer.getChannelData(0));
    const prevTeardown = teardown;
    teardown = () => {
      try {
        proc.disconnect();
      } catch {
        /* ignore */
      }
      try {
        src.disconnect();
      } catch {
        /* ignore */
      }
      prevTeardown();
    };
  }
  return {
    stop: async () => {
      done = true;
      teardown();
      const pcm = encodePcm16(chunks);
      await ctx.close();
      return { pcm, sr };
    },
    cancel: () => {
      done = true;
      teardown();
      void ctx.close();
    },
  };
}

/** Record mono 16k PCM for `secs` seconds via getUserMedia. */
export async function recordSecs(secs: number, onLevel?: (rms: number) => void): Promise<{ pcm: Uint8Array; sr: number }> {
  const rec = await createRecorder(onLevel);
  await new Promise((r) => setTimeout(r, Math.max(1, secs) * 1000));
  return rec.stop();
}

export type VoiceCapture = {
  /** Raw mic stream (STT bytes come from `snapshot`, not this). */
  raw: MediaStream;
  /** Mic (+ later TTS taps) mixed for the beam. Release disconnects the tap. */
  mixed: MediaStream;
  releaseMix: () => void;
  /** PCM16 mono of everything captured so far. */
  snapshotPcm: () => { pcm: Uint8Array; sr: number };
  stop: () => void;
};

/** Voice-mode capture: one mic + one context driving PCM chunks (STT
    stream), live levels, and the beam mix bus together. */
export async function startVoiceCapture(opts?: {
  onLevel?: (rms: number) => void;
  onChunk?: (pcm: Float32Array) => void;
}): Promise<VoiceCapture> {
  const sr = 16000;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  const ctx = audioCtx(sr);
  const src = ctx.createMediaStreamSource(stream);
  const chunks: Float32Array[] = [];
  let done = false;
  const feed = (ch: Float32Array) => {
    if (done) return;
    const copy = new Float32Array(ch);
    chunks.push(copy);
    if (opts?.onLevel) {
      let sum = 0;
      for (let i = 0; i < copy.length; i += 8) sum += (copy[i] ?? 0) ** 2;
      opts.onLevel(Math.min(1, Math.sqrt(sum / Math.ceil(copy.length / 8)) * 4));
    }
    try {
      opts?.onChunk?.(copy);
    } catch {
      /* streaming tap is best-effort */
    }
  };
  const disposers: (() => void)[] = [
    () => stream.getTracks().forEach((t) => {
      try {
        t.stop();
      } catch {
        /* ignore */
      }
    }),
  ];
  const onDisposer = (fn: () => void) => disposers.push(fn);
  try {
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, "voicert-recorder", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (ev: MessageEvent) => feed(ev.data as Float32Array);
    src.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    onDisposer(() => {
      try {
        node.port.onmessage = null;
        src.disconnect();
        node.disconnect();
        mute.disconnect();
      } catch {
        /* ignore */
      }
    });
  } catch {
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    src.connect(proc);
    proc.connect(ctx.destination);
    proc.onaudioprocess = (ev) => feed(ev.inputBuffer.getChannelData(0));
    onDisposer(() => {
      try {
        proc.disconnect();
      } catch {
        /* ignore */
      }
      try {
        src.disconnect();
      } catch {
        /* ignore */
      }
    });
  }
  const mix = voiceMixStream(stream);
  const stop = () => {
    done = true;
    try {
      mix.release();
    } catch {
      /* ignore */
    }
    for (const fn of disposers.splice(0)) {
      try {
        fn();
      } catch {
        /* ignore */
      }
    }
    void ctx.close();
  };
  return {
    raw: stream,
    mixed: mix.mixed,
    releaseMix: () => {
      try {
        mix.release();
      } catch {
        /* ignore */
      }
    },
    snapshotPcm: () => ({ pcm: encodePcm16(chunks), sr }),
    stop,
  };
}

/** Play float32 mono samples at `sr`. Resolves when done. */
export async function playF32(samples: Float32Array, sr: number): Promise<void> {
  const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctx({ sampleRate: sr });
  try {
    // Contexts created after an await (e.g. post-fetch) can start
    // suspended under autoplay policy — resume before playing.
    await ctx.resume();
  } catch {
    /* ignore */
  }
  const buf = ctx.createBuffer(1, samples.length, sr);
  buf.getChannelData(0).set(samples);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  await new Promise<void>((resolve) => {
    src.onended = () => {
      void ctx.close().then(() => resolve());
    };
  });
}

// Shared voice-mix bus: mic + assistant TTS tapped together so the beam
// dances to both sides of the conversation. The destination stream is only
// analysed, never played — silent unless something reads it.
let mixCtx: AudioContext | null = null;
let mixDest: MediaStreamAudioDestinationNode | null = null;

function ensureMix(): { ctx: AudioContext; dest: MediaStreamAudioDestinationNode } | null {
  try {
    if (!mixCtx || !mixDest) {
      mixCtx = audioCtx(48000);
      mixDest = mixCtx.createMediaStreamDestination();
    }
    return { ctx: mixCtx, dest: mixDest };
  } catch {
    return null;
  }
}

/** Mix a mic stream into the shared bus. Release disconnects the tap. */
export function voiceMixStream(mic: MediaStream): { mixed: MediaStream; release: () => void } {
  const mix = ensureMix();
  if (!mix) return { mixed: mic, release: () => undefined };
  const src = mix.ctx.createMediaStreamSource(mic);
  try {
    src.connect(mix.dest);
  } catch {
    return { mixed: mic, release: () => undefined };
  }
  return {
    mixed: mix.dest.stream,
    release: () => {
      try {
        src.disconnect();
      } catch {
        /* ignore */
      }
    },
  };
}

/** Create (or wake) the mix bus inside a user gesture so later taps run. */
export function primeVoiceMix(): void {
  try {
    const mix = ensureMix();
    if (mix && mix.ctx.state === "suspended") void mix.ctx.resume();
  } catch {
    /* ignore */
  }
}

/** Tap assistant TTS into the shared bus (assistant voice in the beam). */
export function tapTtsIntoMix(samples: Float32Array, sr: number): void {
  try {
    const mix = ensureMix();
    if (!mix) return;
    const targetSr = mix.ctx.sampleRate;
    let data = samples;
    if (sr !== targetSr && sr > 0) {
      const ratio = targetSr / sr;
      const out = new Float32Array(Math.max(1, Math.ceil(samples.length * ratio)));
      for (let i = 0; i < out.length; i++) out[i] = samples[Math.min(samples.length - 1, Math.floor(i / ratio))] ?? 0;
      data = out;
    }
    const buf = mix.ctx.createBuffer(1, data.length, targetSr);
    buf.getChannelData(0).set(data);
    const node = mix.ctx.createBufferSource();
    node.buffer = buf;
    node.connect(mix.dest);
    node.start();
  } catch {
    /* beam tap is best-effort */
  }
}
