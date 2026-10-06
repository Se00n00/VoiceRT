// WebAudio mic capture + TTS playback. Replaces arecord/aplay subprocesses.

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

/** Controllable mic recorder: stop() whenever the user is done (dictation UI). */
export async function createRecorder(onLevel?: (rms: number) => void): Promise<Recorder> {
  const sr = 16000;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  const ctx = audioCtx(sr);
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  let done = false;
  const teardown = () => {
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
    stream.getTracks().forEach((t) => t.stop());
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  proc.onaudioprocess = (ev) => {
    if (done) return;
    const ch = ev.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(ch));
    if (onLevel) {
      let sum = 0;
      for (let i = 0; i < ch.length; i += 8) sum += (ch[i] ?? 0) ** 2;
      onLevel(Math.min(1, Math.sqrt(sum / Math.ceil(ch.length / 8)) * 4));
    }
  };
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

/** Play float32 mono samples at `sr`. Resolves when done. */
export async function playF32(samples: Float32Array, sr: number): Promise<void> {
  const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctx({ sampleRate: sr });
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
