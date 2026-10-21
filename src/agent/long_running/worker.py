"""Worker leg lifecycle: warm once, latch failures, report a reason.

The runner asks ``ensure`` before a turn; a refused warm latches so later
turns fail fast with the same reason. ``down_notice`` is the shared
transcript/reply shape for a turn the worker could not run.
"""

from src.agent.prompts.voice import WORKER_DOWN_TEMPLATE

__all__ = ["WorkerLeg"]


class WorkerLeg:
    """One model worker's warm state and failure report."""

    def __init__(self, get_llm, get_chat_model, eager: bool = False):
        """Store live leg getters; warms nothing."""
        self._get_llm = get_llm
        self._get_chat_model = get_chat_model
        self.eager = bool(eager)
        self.warmed = False
        self.unavailable: str | None = None
        self.missing: list = []

    async def ensure(self) -> str:
        """Warm the worker on first use; "" when usable.

        A refusal latches so later turns fail fast with the same reason.
        ..
        """
        if self.unavailable:
            return str(self.unavailable)
        if self.warmed:
            return ""
        llm = self._get_llm()
        if llm is None:
            if self._get_chat_model() is not None:
                self.warmed = True
                return ""
            reason = "no worker leg constructed"
            self.unavailable = reason
            return reason
        try:
            await llm.warm()
        except Exception as exc:  # noqa: BLE001 - latch, report, move on
            reason = str(exc)[:200] or type(exc).__name__
            self.unavailable = reason
            try:
                self.missing.append(f"llm leg: {reason}")
            except Exception:
                pass
            return reason
        self.warmed = True
        return ""

    def missing_reason(self) -> str:
        """Why the worker is unusable this process, or "" if fine.

        ..
        """
        if self.unavailable:
            return str(self.unavailable)[:200]
        for entry in list(self.missing or []):
            text = str(entry)
            if text.startswith("llm leg:"):
                return text.split(":", 1)[1].strip()[:200]
        return ""

    @staticmethod
    def down_notice(task: str, reason: str) -> tuple:
        """Transcript entry for a turn the worker could not run.

        ..
        """
        return (
            str(task),
            WORKER_DOWN_TEMPLATE.format(reason=reason),
            "",
        )
