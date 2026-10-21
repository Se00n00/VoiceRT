"""Planner -> specialist handoff: one model, three simulated subagents.

Declarative deepagents ``SubAgent`` specs (shell-use / browser-use /
computer-use) that all bind the SAME warmed chat model — the simulation:
specialization lives in prompt + tag-filtered tools + middleware, never in
extra weights. ``LongRunningAgent.build_longrunning_agent`` passes the
list to ``create_deep_agent(subagents=...)``, whose ``task`` tool is
the handoff transport: down goes the task description only (isolated
mode), up comes the specialist's final text as a ToolMessage.

Contracts: spec middleware is not inherited from the parent (deepagents
forks parent middleware only under ``mode="fork"``), so policy, trimming
and the read-only filesystem cap are declared here explicitly. Recursion
is impossible by construction: a subagent stack has no ``task`` tool.

Dependencies: deepagents middleware, the repo policy/trim middleware, the
tag table in ``src.agent.mcp.specs``, and the prompt data in
``src.agent.long_running.prompts.handoff``. Called once per turn-build.
"""
from deepagents.middleware.filesystem import FilesystemMiddleware

from src.agent.tool_router import build_execution_tools
from src.agent.long_running.middleware.policy import PolicyMiddleware
from src.agent.long_running.middleware.trim import TrimObservationsMiddleware
from src.agent.long_running.prompts.handoff import (
    SPECIALIST_DESCRIPTIONS,
    SPECIALIST_SLIVER,
    SPECIALIST_WARNINGS,
)

__all__ = ["ROLE_TAGS", "READ_ONLY_FS_TOOLS", "build_specialists"]

ROLE_TAGS = {
    "shell-use": "ShellUse",
    "browser-use": "BrowserUse",
    "computer-use": "ComputerUse",
}
"""Specialist name -> ``TOOL_TAGS`` group that forms its toolset."""

READ_ONLY_FS_TOOLS = ["ls", "read_file", "glob", "grep"]
"""Filesystem allowlist for non-shell specialists (no execute, no writes)."""


def build_specialists(model, pool, *, backend=None, confirmer=None,
                      limit: int = 1500) -> list:
    """Build the three specialist SubAgent specs over one shared model.

    ``pool`` is the full execution-tool pool; each specialist is bound to
    its tag group only. ``confirmer`` is the per-turn approval callback —
    specialists run the identical policy gate as the planner. ``limit``
    caps observation size so a specialist cannot flood its own window.
    ..
    """
    specs = []
    for name, tag in ROLE_TAGS.items():
        middleware = [
            TrimObservationsMiddleware(limit=limit),
            PolicyMiddleware(confirmer=confirmer),
        ]
        if tag != "ShellUse":
            middleware.insert(0, FilesystemMiddleware(
                backend=backend, tools=list(READ_ONLY_FS_TOOLS)))
        specs.append({
            "name": name,
            "description": SPECIALIST_DESCRIPTIONS[name],
            "system_prompt": SPECIALIST_SLIVER % (
                name.split("-")[0], SPECIALIST_WARNINGS.get(name, "")),
            "tools": build_execution_tools(pool, tags={tag}),
            "model": model,
            "middleware": middleware,
        })
    return specs
