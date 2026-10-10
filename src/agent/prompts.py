"""Agent prompts. Backward-compatible shim over :mod:`src.prompts`.

``TERMINAL_PREAMBLE`` now lives in :mod:`src.prompts.terminal`; this
module re-exports it so existing imports keep working.
"""

from src.prompts.terminal import TERMINAL_PREAMBLE  # noqa: F401

__all__ = ["TERMINAL_PREAMBLE"]
