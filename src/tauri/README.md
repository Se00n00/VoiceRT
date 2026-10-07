# VoiceRT — Tauri GUI (replaces `src/tui`)

Fresh rewrite with Tailwind CSS, from the floating-cards-over-wallpaper template:
black icon rail left,chat transcript + composer center, Get Started / model /
Copy&Run cards right.

Pure client by design: it **never spawns** `bridge.py`. Run the model backend
first, then point the UI at it.

```bash
# 1) backend (one GPU process at a time)
PYTHONPATH=. .venv/bin/python -m src.server  # :8004

# 2) UI
cd src/tauri && npm install && npm run dev
# optional: VITE_VOICE_BRIDGE=http://127.0.0.1:8004 npm run dev

# 3) desktop shell (needs Rust + WebKitGTK, not on this box yet)
cd src/tauri && tauri dev        # or: tauri build
```

Checks: `npm run typecheck`, `npm run build`.
