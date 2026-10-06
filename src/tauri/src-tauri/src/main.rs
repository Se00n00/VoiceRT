// VoiceRT Tauri shell: pure client window. No sidecar, no Python spawn.
// Start bridge.py separately: `PYTHONPATH=. .venv/bin/python bridge.py`
// The frontend talks to it at VITE_VOICE_BRIDGE (default 127.0.0.1:8004).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
