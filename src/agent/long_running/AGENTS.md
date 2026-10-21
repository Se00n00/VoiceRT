# Long-running agent — standing instructions

This file is appended to the agent's core system prompt on every build.
Edit it to add machine-specific or personal standing instructions
(paths, conventions, preferences). The core prompt itself stays fixed
in `src/agent/long_running/prompts/core.py`.

- Keep entries short and imperative.
- Read at build time, so changes apply on the next turn.
