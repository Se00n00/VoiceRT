import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [solid(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    target: "es2021",
    // Never inline the AudioWorklet as a data: URL — it must load as a
    // same-origin file module inside the WebView.
    assetsInlineLimit: 0,
  },
});
