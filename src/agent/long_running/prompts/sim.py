"""Legacy Sim-Agent (single-agent subagents) templates.

The lean worker sliver, the plan prompt, and the subtask framing /
compose templates used by ``src.agent.long_running.simtask``. The native
deepagents handoff lives in ``src.agent.long_running.prompts.handoff``.
"""

__all__ = [
    "SYSTEM_SLIVER",
    "PLAN_PROMPT",
    "MAX_SUBTASKS",
    "SUBTASK_HEAD_TEMPLATE",
    "FINAL_COMPOSE_TEMPLATE",
    "SUBTASK_ENVELOPE_TEMPLATE",
]

MAX_SUBTASKS = 6
"""Mirrors max_agent_steps: bounded by construction."""

SYSTEM_SLIVER = """You are a focused worker. Parent goal: %s. Do exactly the assigned subtask: use the available tools when they help, then reply with the result only when done."""

PLAN_PROMPT = """Break this request into numbered subtasks (one per line, `1. ...`). Reply with ONLY the numbered list, no tools, no preamble. If it needs fewer than 2 subtasks, reply with exactly one line. Request: %s"""

SUBTASK_HEAD_TEMPLATE = """Goal: subtask %d of %d: %s.
Parent goal: %s. Complete ONLY this subtask.

Return Format: when done, reply with one short paragraph: what you did + file paths/values produced. No envelopes, no commentary.

Warnings: mutating commands need an explicit user yes; destructive commands (rm -rf /, mkfs, fork-bombs, dd to devices) are blocked — never emit them. Use only these ops: %s. Do NOT plan further subtasks (depth 1 only) — execute flat.

--

Context Dump (the only parent context you get):
%s"""

FINAL_COMPOSE_TEMPLATE = """Answer the original request from these subtask results (no more tools needed unless missing something): %s"""

SUBTASK_ENVELOPE_TEMPLATE = """▸ subtask %d/%d [%s]: %s"""
