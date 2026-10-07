// Solid port of cult-ui's `ai-blob-warp` avatar (MIT).
// Circular frame with Paper Design's animated `Warp` shader inside
// (@paper-design/shaders, Apache-2.0 — same defaults as the cult component).
// Motion-safe: the pulse is transform-only CSS, the shader freezes when
// `prefers-reduced-motion` is set and auto-pauses off-screen via
// ShaderMount's own IntersectionObserver.
import { onCleanup, onMount } from "solid-js";
import {
  ShaderMount,
  getShaderColorFromString,
  getShaderNoiseTexture,
  ShaderFitOptions,
  warpFragmentShader,
  type ShaderMountUniforms,
} from "@paper-design/shaders";

/** Blue stops matching the voice blob gradient. */
const WARP_COLORS = ["#7dd3fc", "#38bdf8", "#2563eb", "#1e3a8a"];

export function WarpOrb() {
  let host!: HTMLDivElement;
  let mount: ShaderMount | undefined;
  let cancelled = false;

  onMount(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const start = () => {
      if (cancelled || mount) return;
      const noise = getShaderNoiseTexture();
      const uniforms: ShaderMountUniforms = {
        u_colors: WARP_COLORS.map((c) => getShaderColorFromString(c)),
        u_colorsCount: WARP_COLORS.length,
        u_proportion: 0.54,
        u_softness: 1,
        u_shape: 0, // checks (cult default)
        u_shapeScale: 1,
        u_distortion: 0.25,
        u_swirl: 0.8,
        u_swirlIterations: 10,
        u_fit: ShaderFitOptions.none,
        u_scale: 0.2,
        u_rotation: 0,
        u_originX: 0.5,
        u_originY: 0.5,
        u_offsetX: 0,
        u_offsetY: 0,
        u_worldWidth: 1280,
        u_worldHeight: 720,
      };
      // The mount throws on an incomplete image, so only attach a decoded one.
      if (noise && noise.complete && noise.naturalWidth > 0) {
        uniforms.u_noiseTexture = noise;
      }
      mount = new ShaderMount(host, warpFragmentShader, uniforms, undefined, mq.matches ? 0 : 1);
    };
    // The noise texture is a data-URL image: mount once it's decoded.
    const probe = getShaderNoiseTexture();
    if (probe && !probe.complete) {
      probe.addEventListener("load", start, { once: true });
    } else {
      start();
    }
    const onChange = () => mount?.setSpeed(mq.matches ? 0 : 1);
    mq.addEventListener("change", onChange);
    onCleanup(() => {
      cancelled = true;
      mq.removeEventListener("change", onChange);
      mount?.dispose();
      mount = undefined;
    });
  });

  return (
    <div class="warp-pulse pointer-events-none absolute inset-0" aria-hidden="true">
      <div ref={host} class="h-full w-full" />
    </div>
  );
}
