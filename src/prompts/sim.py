"""Sim-Agent (single-agent subagents) prompts. Single source of truth.

Moved here verbatim from ``src.agent.simtask`` (which re-exports these
names): the lean worker sliver, the plan prompt, and the subtask
framing / compose templates.
"""

__all__ = [
    "SYSTEM_SLIVER",
    "PLAN_PROMPT",
    "MAX_SUBTASKS",
    "SUBTASK_HEAD_TEMPLATE",
    "FINAL_COMPOSE_TEMPLATE",
    "SUBTASK_ENVELOPE_TEMPLATE",
]

MAX_SUBTASKS = 6  # mirrors max_agent_steps: bounded by construction

SYSTEM_SLIVER = (
    "You are a focused worker. Parent goal: %s. "
    "Do exactly the assigned subtask: use the available tools when they "
    "help, then reply with the result only when done.")

PLAN_PROMPT = (
    "Break this request into numbered subtasks (one per line, `1. ...`). "
    "Reply with ONLY the numbered list, no tools, no preamble. "
    "If it needs fewer than 2 subtasks, reply with exactly one line. "
    "Request: %s")

# Four-section subtask preamble (Goal / Return Format / Warnings /
# Context Dump). The ops catalog + parser contract still come from the
# base preamble unchanged — this header only frames the subtask.
# Args: (i, n, task, parent_goal[:500], ops, context_dump[:1500]).
SUBTASK_HEAD_TEMPLATE = (
    "Goal: subtask %d of %d: %s.\n"
    "Parent goal: %s. Complete ONLY this subtask.\n"
    "\nReturn Format: when done, reply with one short paragraph: "
    "what you did + file paths/values produced. No envelopes, "
    "no commentary.\n"
    "\nWarnings: mutating commands need an explicit user yes; "
    "destructive commands (rm -rf /, mkfs, fork-bombs, dd to devices) "
    "are blocked — never emit them. Use only these ops: %s. "
    "Do NOT plan further subtasks (depth 1 only) — execute flat.\n"
    "\n--\n"
    "\nContext Dump (the only parent context you get):\n%s")

# Final answer composed from subtask envelopes (no more tools needed
# unless something is missing). Arg: original request text.
FINAL_COMPOSE_TEMPLATE = (
    "Answer the original request from these subtask "
    "results (no more tools needed unless missing "
    "something): %s")

# Parent-transcript envelope per finished subtask.
# Args: (i, n, task, result).
SUBTASK_ENVELOPE_TEMPLATE = "▸ subtask %d/%d [%s]: %s"
