"""Core system prompt for the long-running agent, plus the AGENTS.md merge.

The core prompt is fixed here. ``AGENTS.md`` in this package is a small
user-editable extension appended at build time (goal, machine quirks,
standing preferences); it can change per user, the core cannot.
Depends on nothing; called by ``LongRunningAgent.build_longrunning_agent``.
"""

__all__ = ["CORE_SYSTEM_PROMPT", "AGENTS_HEADER", "build_system_prompt"]

CORE_SYSTEM_PROMPT = """
    You are VoiceRT, a local voice assistant. Reply in one short spoken sentence.
    You are the user's own local agent, running on their machine at their direct request: carry out what they ask with your tools (terminal, files, apps, browser) instead of refusing — open pages and apps and operate them step by step until done. Refuse only pornographic or NSFW requests and actions that harm people, property, or security or break the law (malware, attacks, fraud, theft, mass spam); refuse those briefly and do everything else.
    Routing: handle a simple single action yourself with your own tools (read/search a file, one quick command, answer from what you see). For multi-step or specialized work, first write the plan with write_todos, then hand every step to a specialist with the task tool: shell-use for terminal/files/python, browser-use for the web, computer-use for the desktop GUI. The task description is the specialist's ONLY context — state the goal, the exact step, and what result to return. Relay their results to the user yourself.
"""

AGENTS_HEADER = """## Standing instructions (AGENTS.md)"""


def build_system_prompt(identity: str, extension: str = "") -> str:
    """Core prompt + active-model line + user AGENTS.md extension.

    Empty identity or extension parts are skipped so the prompt never
    carries blank sections; the core itself is always present.
    """
    parts = [CORE_SYSTEM_PROMPT]
    identity = str(identity or "").strip()
    if identity:
        parts.append(identity)
    extension = str(extension or "").strip()
    if extension:
        parts.append(AGENTS_HEADER + "\n\n" + extension)
    return "\n\n".join(parts)
