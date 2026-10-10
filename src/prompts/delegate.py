"""Two-brain delegation prompts. Single source of truth.

Moved here verbatim from ``src.agent.delegate`` (which re-exports these
names). The boolean contract, its few-shot pairs, its token budget, and
the retired ``delegate`` tool spec all live here.
"""

__all__ = [
    "BOOLEAN_SYSTEM",
    "BOOLEAN_FEWSHOT",
    "BOOLEAN_MAX_TOKENS",
    "DELEGATE_TOOL",
    "DELEGATE_TOOL_NAME",
]

DELEGATE_TOOL_NAME = "delegate"

# The front model's ONLY job is to raise a flag. See src.agent.delegate
# for the full rationale (0/8 valid calls with the old tool envelope,
# 19/19 tasks escalate with the boolean).
BOOLEAN_SYSTEM = (
    "Answer YES only if the user wants you to actually DO something: run a "
    "command, read or change a file, install, search, look up, fetch, or "
    "operate on their computer. Answer NO for greetings, opinions, jokes, "
    "thanks, general knowledge, and questions about code or concepts you "
    "can explain from memory.")

# Few-shot pairs, not decoration: the same prompt without them answered
# YES to 100% of chit-chat and NO to half the real tasks. These five were
# the difference between 22/37 and 34/37.
BOOLEAN_FEWSHOT = [
    {"role": "user", "content": "USER: list the files in the current directory\nANSWER:"},
    {"role": "assistant", "content": "YES"},
    {"role": "user", "content": "USER: hi\nANSWER:"},
    {"role": "assistant", "content": "NO"},
    {"role": "user", "content": "USER: why is my code so slow\nANSWER:"},
    {"role": "assistant", "content": "YES"},
    {"role": "user", "content": "USER: ok cool\nANSWER:"},
    {"role": "assistant", "content": "NO"},
    {"role": "user", "content": "USER: summarize this article\nANSWER:"},
    {"role": "assistant", "content": "YES"},
]

# 6 tokens is the whole budget: the answer is one word, so anything longer
# is the model ignoring the format rather than thinking.
BOOLEAN_MAX_TOKENS = 6

# RETIRED — kept only so external callers and the historical tests do not
# break. The front leg is no longer given this tool; it gets no schema at
# all. House flat shape ({"name", ...}), like TERMINAL_TOOLS.
DELEGATE_TOOL = {
    "name": DELEGATE_TOOL_NAME,
    "description": (
        "Hand a task to the worker agent and let it do the work. The worker "
        "has real tools: a shell, file read/write/edit, code search, a web "
        "fetch and search, Python, and its own subagents. Call this for "
        "ANY action on this machine or the web - reading or changing files, "
        "running commands, looking something up in the repo, installing, "
        "debugging, checking status. Reply with plain text and do NOT call "
        "this for small talk, opinions, or questions you can answer from "
        "your own knowledge. Never describe the work as done: either call "
        "this tool, or say nothing happened."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The request in full, as the worker should carry it out. "
                    "Carry over every detail the user gave: file paths, "
                    "names, constraints, and what 'done' should look like."
                ),
            }
        },
        "required": ["task"],
    },
}
