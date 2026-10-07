import { createEffect, onCleanup, onMount } from "solid-js";
import * as THREE from "three";

// Chat panels cleared per frame (transcript only, when a conversation
// exists) with the same flatten-to-background as the mouse hover.
const MAX_ERASE = 8;

// Exact wave shaders from the react-bits Dither snippet, plus one additive
// `scrollPhase` uniform (wired to the transcript scroll = layer 3 parallax).
const waveVertexShader = `
precision highp float;
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 modelPosition = modelMatrix * vec4(position, 1.0);
  vec4 viewPosition = viewMatrix * modelPosition;
  gl_Position = projectionMatrix * viewPosition;
}
`;

const waveFragmentBase = `
precision highp float;
uniform vec2 resolution;
uniform float time;
uniform float waveSpeed;
uniform float waveFrequency;
uniform float waveAmplitude;
uniform vec3 waveColor;
uniform vec3 backgroundColor;
uniform vec2 mousePos;
uniform int enableMouseInteraction;
uniform float mouseRadius;
// Conversation erase: same flatten-to-background as the mouse hover, pinned
// to the transcript panel rect so chat sits on clean ground.
uniform vec4 uErase[${MAX_ERASE}];
uniform int uEraseCount;
uniform float uEraseRadius;
uniform float uEraseFeather;

vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec2 fade(vec2 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }

float cnoise(vec2 P) {
  vec4 Pi = floor(P.xyxy) + vec4(0.0,0.0,1.0,1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0,0.0,1.0,1.0);
  Pi = mod289(Pi);
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = fract(i * (1.0/41.0)) * 2.0 - 1.0;
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x, gy.x);
  vec2 g10 = vec2(gx.y, gy.y);
  vec2 g01 = vec2(gx.z, gy.z);
  vec2 g11 = vec2(gx.w, gy.w);
  vec4 norm = taylorInvSqrt(vec4(dot(g00,g00), dot(g01,g01), dot(g10,g10), dot(g11,g11)));
  g00 *= norm.x; g01 *= norm.y; g10 *= norm.z; g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  return 2.3 * mix(n_x.x, n_x.y, fade_xy.y);
}

const int OCTAVES = 4;
float fbm(vec2 p) {
  float value = 0.0;
  float amp = 1.0;
  float freq = waveFrequency;
  for (int i = 0; i < OCTAVES; i++) {
    value += amp * abs(cnoise(p));
    p *= freq;
    amp *= waveAmplitude;
  }
  return value;
}

float pattern(vec2 p) {
  vec2 p2 = p - time * waveSpeed;
  return fbm(p + fbm(p2));
}

// Rounded-box SDF (uv space, y-up, x aspect-scaled like the mouse math).
float sdRoundBox(vec2 p, vec2 c, vec2 b, float r) {
  vec2 q = abs(p - c) - b + r;
  return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  uv -= 0.5;
  uv.x *= resolution.x / resolution.y;
  float f = pattern(uv);
  if (enableMouseInteraction == 1) {
    vec2 mouseNDC = (mousePos / resolution - 0.5) * vec2(1.0, -1.0);
    mouseNDC.x *= resolution.x / resolution.y;
    float dist = length(uv - mouseNDC);
    float effect = 1.0 - smoothstep(0.0, mouseRadius, dist);
    f -= 0.5 * effect;
  }
  // Same erase pinned to the conversation panel: fully clean inside, with
  // a wide feathered border where the dither escapes back in. The left
  // border gets a wider + weaker erase so no straight edge survives there.
  for (int i = 0; i < ${MAX_ERASE}; i++) {
    if (i < uEraseCount) {
      vec4 e = uErase[i];
      float d = sdRoundBox(uv, e.xy, e.zw, uEraseRadius);
      float leftness = clamp((e.x - uv.x) / max(1e-4, e.z + uEraseFeather), 0.0, 1.0);
      leftness = leftness * leftness * (3.0 - 2.0 * leftness);
      float feather = mix(uEraseFeather, uEraseFeather * 2.6, leftness);
      float strength = mix(0.5, 0.3, leftness);
      float eff = 1.0 - smoothstep(0.0, feather, d);
      f -= strength * eff;
    }
  }
  vec3 col = mix(backgroundColor, waveColor, clamp(f, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}
`;

const waveFragmentShader = waveFragmentBase
  .replace("uniform float mouseRadius;", "uniform float mouseRadius;\nuniform float scrollPhase;")
  .replace("vec2 p2 = p - time * waveSpeed;", "vec2 p2 = p - time * waveSpeed - scrollPhase;");

// Bayer dither pass (same matrix/threshold math as the snippet), standalone:
// samples the wave pass through tDither instead of postprocessing's inputBuffer.
const ditherVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ditherFragmentShader = `
precision highp float;
uniform sampler2D tDither;
uniform vec2 resolution;
uniform float colorNum;
uniform float pixelSize;
varying vec2 vUv;

const float bayerMatrix8x8[64] = float[64](
  0.0/64.0, 48.0/64.0, 12.0/64.0, 60.0/64.0,  3.0/64.0, 51.0/64.0, 15.0/64.0, 63.0/64.0,
  32.0/64.0,16.0/64.0, 44.0/64.0, 28.0/64.0, 35.0/64.0,19.0/64.0, 47.0/64.0, 31.0/64.0,
  8.0/64.0, 56.0/64.0,  4.0/64.0, 52.0/64.0, 11.0/64.0,59.0/64.0,  7.0/64.0, 55.0/64.0,
  40.0/64.0,24.0/64.0, 36.0/64.0, 20.0/64.0, 43.0/64.0,27.0/64.0, 39.0/64.0, 23.0/64.0,
  2.0/64.0, 50.0/64.0, 14.0/64.0, 62.0/64.0,  1.0/64.0,49.0/64.0, 13.0/64.0, 61.0/64.0,
  34.0/64.0,18.0/64.0, 46.0/64.0, 30.0/64.0, 33.0/64.0,17.0/64.0, 45.0/64.0, 29.0/64.0,
  10.0/64.0,58.0/64.0,  6.0/64.0, 54.0/64.0,  9.0/64.0,57.0/64.0,  5.0/64.0, 53.0/64.0,
  42.0/64.0,26.0/64.0, 38.0/64.0, 22.0/64.0, 41.0/64.0,25.0/64.0, 37.0/64.0, 21.0/64.0
);

vec3 dither(vec2 uv, vec3 color) {
  vec2 scaledCoord = floor(uv * resolution / pixelSize);
  int x = int(mod(scaledCoord.x, 8.0));
  int y = int(mod(scaledCoord.y, 8.0));
  float threshold = bayerMatrix8x8[y * 8 + x] - 0.25;
  float step = 1.0 / (colorNum - 1.0);
  color += threshold * step;
  float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float bias = mix(0.2, 0.0, smoothstep(0.45, 0.8, luminance));
  color = clamp(color - bias, 0.0, 1.0);
  return floor(color * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
}

void main() {
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(vUv / normalizedPixelSize);
  vec4 color = texture2D(tDither, uvPixel);
  color.rgb = dither(vUv, color.rgb);
  gl_FragColor = color;
}
`;

/**
 * Three.js port of the react-bits Dither: fbm wave field (layer 1+2) rendered
 * to a target, then the 8x8 Bayer dither pass. Transcript scroll feeds
 * `scrollPhase`, so scrolling reshapes the clouds (layer 3).
 */
export function DitherBg(props: { scrollId: string; light: boolean }) {
  let canvas!: HTMLCanvasElement;

  onMount(() => {
    const parent = canvas.parentElement;
    if (!parent) return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);

    const waveUniforms: Record<string, THREE.IUniform> = {
      time: { value: 0 },
      resolution: { value: new THREE.Vector2(1, 1) },
      waveSpeed: { value: 0.015 },
      waveFrequency: { value: 3 },
      waveAmplitude: { value: 0.3 },
      waveColor: { value: new THREE.Color(0.5, 0.5, 0.5) },
      backgroundColor: { value: new THREE.Color(0, 0, 0) },
      mousePos: { value: new THREE.Vector2(0, 0) },
      enableMouseInteraction: { value: 1 },
      mouseRadius: { value: 0.3 },
      scrollPhase: { value: 0 },
      // Conversation erase rects: xy = center, zw = half extents (uv space,
      // same NDC mapping as the mouse). Degenerate (-1 half-size) = unused.
      uErase: {
        value: Array.from({ length: MAX_ERASE }, () => new THREE.Vector4(0, 0, -1, -1)),
      },
      uEraseCount: { value: 0 },
      uEraseRadius: { value: 0.03 },
      uEraseFeather: { value: 0.08 },
    };

    const ditherUniforms: Record<string, THREE.IUniform> = {
      tDither: { value: null as THREE.Texture | null },
      resolution: { value: new THREE.Vector2(1, 1) },
      colorNum: { value: 4 },
      pixelSize: { value: 2 },
    };

    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quad = new THREE.PlaneGeometry(2, 2);

    const waveScene = new THREE.Scene();
    waveScene.add(
      new THREE.Mesh(quad, new THREE.ShaderMaterial({ vertexShader: waveVertexShader, fragmentShader: waveFragmentShader, uniforms: waveUniforms })),
    );

    const ditherScene = new THREE.Scene();
    ditherScene.add(
      new THREE.Mesh(quad, new THREE.ShaderMaterial({ vertexShader: ditherVertexShader, fragmentShader: ditherFragmentShader, uniforms: ditherUniforms })),
    );

    let rt = new THREE.WebGLRenderTarget(2, 2);
    ditherUniforms.tDither!.value = rt.texture;

    // light theme: pale field + steel wave (dark theme stays black/gray)
    createEffect(() => {
      const light = props.light;
      (waveUniforms.backgroundColor!.value as THREE.Color).setRGB(
        light ? 0.87 : 0,
        light ? 0.89 : 0,
        light ? 0.94 : 0,
      );
      (waveUniforms.waveColor!.value as THREE.Color).setRGB(
        light ? 0.42 : 0.5,
        light ? 0.48 : 0.5,
        light ? 0.66 : 0.5,
      );
    });

    const resize = () => {
      const r = parent.getBoundingClientRect();
      const w = Math.max(2, Math.floor(r.width));
      const h = Math.max(2, Math.floor(r.height));
      renderer.setSize(w, h, false);
      rt.setSize(w, h);
      (waveUniforms.resolution!.value as THREE.Vector2).set(w, h);
      (ditherUniforms.resolution!.value as THREE.Vector2).set(w, h);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(parent);

    const onPointer = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      (waveUniforms.mousePos!.value as THREE.Vector2).set(e.clientX - r.left, e.clientY - r.top);
    };
    parent.addEventListener("pointermove", onPointer);

    let scrollTarget = 0;
    let scrollPhase = 0;
    const scroller = document.getElementById(props.scrollId);
    const onScroll = () => {
      scrollTarget = scroller ? scroller.scrollTop * 0.0002 : 0;
    };
    scroller?.addEventListener("scroll", onScroll, { passive: true });

    // Pin the mouse-style erase to every [data-erase] panel (transcript
    // with a real conversation). Element list refreshes periodically;
    // rects re-measure every frame so scrolling stays glued.
    const ERASE_PAD = 96;
    let eraseEls: Element[] = [];
    let eraseTick = 0;
    const updateErase = () => {
      if (eraseTick % 15 === 0) {
        eraseEls = Array.from(document.querySelectorAll("[data-erase]"));
      }
      eraseTick++;
      const canvasRect = canvas.getBoundingClientRect();
      const res = waveUniforms.resolution!.value as THREE.Vector2;
      const W = Math.max(1, res.x);
      const H = Math.max(1, res.y);
      const aspect = W / H;
      const arr = waveUniforms.uErase!.value as THREE.Vector4[];
      let n = 0;
      for (const el of eraseEls) {
        if (n >= MAX_ERASE) break;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        if (r.bottom < canvasRect.top || r.top > canvasRect.bottom || r.right < canvasRect.left || r.left > canvasRect.right) {
          continue;
        }
        const cx = r.left - canvasRect.left + r.width / 2;
        const cy = r.top - canvasRect.top + r.height / 2;
        const v = arr[n]!;
        v.x = (cx / W - 0.5) * aspect;
        v.y = 0.5 - cy / H;
        v.z = ((r.width / 2 + ERASE_PAD) / W) * aspect;
        v.w = (r.height / 2 + ERASE_PAD) / H;
        n++;
      }
      for (let i = n; i < MAX_ERASE; i++) arr[i]!.set(0, 0, -1, -1);
      waveUniforms.uEraseCount!.value = n;
      waveUniforms.uEraseRadius!.value = 20 / H;
      waveUniforms.uEraseFeather!.value = 130 / H;
    };

    const clock = new THREE.Clock();
    let raf = 0;
    const frame = () => {
      scrollPhase += (scrollTarget - scrollPhase) * 0.025;
      waveUniforms.time!.value = clock.getElapsedTime();
      waveUniforms.scrollPhase!.value = scrollPhase;
      updateErase();
      renderer.setRenderTarget(rt);
      renderer.render(waveScene, cam);
      renderer.setRenderTarget(null);
      renderer.render(ditherScene, cam);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    onCleanup(() => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      parent.removeEventListener("pointermove", onPointer);
      scroller?.removeEventListener("scroll", onScroll);
      quad.dispose();
      rt.dispose();
      waveScene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) (m.material as THREE.Material).dispose();
      });
      ditherScene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) (m.material as THREE.Material).dispose();
      });
      renderer.dispose();
    });
  });

  return <canvas ref={canvas} class="pointer-events-none absolute inset-0 z-0 h-full w-full" aria-hidden="true" />;
}
