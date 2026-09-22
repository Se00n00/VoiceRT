"""Tool packages: single-model action schemas + parsing.

No sidecar model, no intent router. The SAME Qwen3-0.6B instance
decides between chat and one action per step by emitting either
plain text (chat) or a single JSON object (action). Terminal schema
lives in :mod:`src.tools.terminal` (the browser schema was removed
with the extension).
"""
from src.tools.terminal import (
    TERMINAL_PREAMBLE,
    TERMINAL_TOOLS,
    TerminalAction,
    check_policy,
    is_denied,
    needs_confirm,
    parse_terminal_action,
    parse_xml_action,
)

__all__ = [
    "ALLOWED_OPS",
    "TERMINAL_PREAMBLE",
    "TERMINAL_TOOLS",
    "TerminalAction",
    "check_policy",
    "is_denied",
    "needs_confirm",
    "parse_terminal_action",
    "parse_xml_action",
]
