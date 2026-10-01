// Node build for the OpenTUI (Solid) TUI.
//
// @opentui/solid ships a Bun plugin, but this box runs Node. The transform is
// the same one that plugin uses: babel-preset-solid (generate: "universal",
// moduleName "@opentui/solid") + @babel/preset-typescript. The extra
// module-resolver redirect pins `solid-js` to its client build, because Node's
// default export condition resolves solid-js to the server (non-reactive)
// build.
import { transformAsync } from "@babel/core";
import ts from "@babel/preset-typescript";
import solid from "babel-preset-solid";
import moduleResolver from "babel-plugin-module-resolver";
import { readdir, readFile, writeFile, mkdir, stat } from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const SRC = path.join(ROOT, "src");
const OUT = path.join(ROOT, "dist");

function resolveSolidRuntime(specifier) {
  if (specifier === "solid-js") return "solid-js/dist/solid.js";
  if (specifier === "solid-js/store") return "solid-js/store/dist/store.js";
  return specifier;
}

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else if (/\.[cm]?[jt]sx?$/.test(entry.name)) out.push(full);
  }
  return out;
}

async function buildFile(file) {
  const code = await readFile(file, "utf8");
  const isJsx = /\.[cm]?[jt]sx$/.test(file);
  const presets = [];
  if (isJsx) presets.push([solid, { moduleName: "@opentui/solid", generate: "universal" }]);
  presets.push([ts, { onlyRemoveTypeImports: false }]);
  const res = await transformAsync(code, {
    filename: file,
    configFile: false,
    babelrc: false,
    presets,
    plugins: [[moduleResolver, { resolvePath: resolveSolidRuntime }]],
    sourceMaps: false,
  });
  const rel = path.relative(SRC, file).replace(/\.[cm]?[jt]sx?$/, ".js");
  const dest = path.join(OUT, rel);
  await mkdir(path.dirname(dest), { recursive: true });
  await writeFile(dest, res.code ?? code);
  return rel;
}

async function buildAll() {
  const files = await walk(SRC);
  const built = [];
  for (const file of files) built.push(await buildFile(file));
  console.log(`built ${built.length} file(s) -> dist/`);
  return built;
}

async function main() {
  await buildAll();
  if (!process.argv.includes("--watch")) return;
  const { watch } = await import("node:fs");
  console.log("watching src/ for changes…");
  let timer = null;
  watch(SRC, { recursive: true }, () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      buildAll().catch((e) => console.error(e));
    }, 80);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
