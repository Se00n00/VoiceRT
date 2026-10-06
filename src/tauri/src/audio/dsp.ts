/* Pure audio DSP ported from src/tui/src/audio.ts (no Node Buffer). */

export const NB = 24;

/** 24 log-scaled spectrum bins 0..1 from float32 mono samples. */
export function spectrum(samples: Float32Array, nb = NB): number[] {
  const n = samples.length;
  if (n === 0) return new Array(nb).fill(0);
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
export function rmsLevelBytes(pcm: Uint8Array): number {
  if (pcm.length < 2) return 0;
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let sum = 0;
  const n = Math.floor(pcm.length / 2);
  for (let i = 0; i < n; i++) {
    const v = view.getInt16(i * 2, true) / 32768;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / n) * 6);
}

/** int16 mono PCM bytes -> float32 samples. */
export function pcmToF32(pcm: Uint8Array): Float32Array {
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  const n = Math.floor(pcm.length / 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

/** base64 float32-LE (bridge format) -> float32 samples. */
export function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const n = Math.floor(buf.length / 4);
  const out = new Float32Array(n);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** base64 helper for uploads (Uint8Array -> b64). */
export function u8ToB64(u8: Uint8Array): string {
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) {
    s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  }
  return btoa(s);
}
