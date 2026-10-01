// Pure audio helpers: spectrum bins + WAV encode/decode. No deps, unit-testable.
export const NB = 24;

/** 24 log-scaled spectrum bins 0..1 from float32 mono samples. */
export function spectrum(samples: Float32Array, nb = NB): number[] {
  const n = samples.length;
  if (n === 0) return new Array(nb).fill(0);
  // Naive DFT magnitudes grouped into log-ish bands (fine at 4 Hz UI rate).
  const mags: number[] = new Array(nb).fill(0);
  const counts: number[] = new Array(nb).fill(0);
  const maxK = Math.min(512, Math.floor(n / 2));
  for (let k = 1; k < maxK; k++) {
    const bin = Math.min(nb - 1, Math.floor((nb * Math.log1p(k)) / Math.log1p(maxK)));
    let re = 0;
    let im = 0;
    const step = (2 * Math.PI * k) / n;
    for (let t = 0; t < n; t += 4) {
      const s = samples[t] ?? 0;
      re += s * Math.cos(step * t);
      im -= s * Math.sin(step * t);
    }
    mags[bin]! += Math.sqrt(re * re + im * im);
    counts[bin]! += 1;
  }
  const vals = mags.map((v, i) => Math.log1p(v / Math.max(1, counts[i]!)));
  const mx = Math.max(...vals, 1e-9);
  return vals.map((v) => Math.round((v / mx) * 100) / 100);
}

/** RMS level 0..1 of int16 PCM bytes. */
export function rmsLevel(pcm: Buffer): number {
  if (pcm.length < 2) return 0;
  let sum = 0;
  const n = Math.floor(pcm.length / 2);
  for (let i = 0; i < n; i++) {
    const v = pcm.readInt16LE(i * 2) / 32768;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / n) * 6);
}

/** int16 mono PCM bytes -> float32 samples. */
export function pcmToF32(pcm: Buffer): Float32Array {
  const n = Math.floor(pcm.length / 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = pcm.readInt16LE(i * 2) / 32768;
  return out;
}

/** float32 mono samples @sr -> 16-bit WAV Buffer (for aplay stdin). */
export function f32ToWav(samples: Float32Array, sr: number): Buffer {
  const data = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i] ?? 0));
    data.writeInt16LE(Math.round(v * 32767), i * 2);
  }
  const head = Buffer.alloc(44);
  head.write("RIFF", 0);
  head.writeUInt32LE(36 + data.length, 4);
  head.write("WAVE", 8);
  head.write("fmt ", 12);
  head.writeUInt32LE(16, 16);
  head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22);
  head.writeUInt32LE(sr, 24);
  head.writeUInt32LE(sr * 2, 28);
  head.writeUInt16LE(2, 32);
  head.writeUInt16LE(16, 34);
  head.write("data", 36);
  head.writeUInt32LE(data.length, 40);
  return Buffer.concat([head, data]);
}

/** base64 float32-LE (bridge format) -> float32 samples. */
export function b64ToF32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  const n = Math.floor(buf.length / 4);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = buf.readFloatLE(i * 4);
  return out;
}

/** Per-frame spectrum envelope for playback animation. */
export function envelope(samples: Float32Array, frames = 60): number[][] {
  const out: number[][] = [];
  const n = samples.length;
  for (let i = 0; i < frames; i++) {
    const seg = samples.slice(Math.floor((i * n) / frames), Math.floor(((i + 1) * n) / frames));
    out.push(spectrum(seg));
  }
  return out;
}
