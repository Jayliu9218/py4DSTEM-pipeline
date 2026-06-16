from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable


ProgressCallback = Callable[[str, float], None]
TaskOperation = Callable[[ProgressCallback], Any]


class ComputationCancelled(RuntimeError):
    """Raised when a cooperative computation task has been cancelled."""


@dataclass
class CancellationToken:
    """Small cooperative cancellation primitive shared by background tasks."""

    _event: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check_cancelled(self) -> None:
        if self.is_cancelled:
            raise ComputationCancelled("Computation cancelled.")


@dataclass(frozen=True)
class ComputationTask:
    """Internal task description for long-running scientific work."""

    name: str
    operation: TaskOperation
    memory_budget_mb: int | None = None
    result_key: str | None = None
    status_message: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    cancel_token: CancellationToken = field(default_factory=CancellationToken)

    @property
    def memory_budget_bytes(self) -> int | None:
        if self.memory_budget_mb is None:
            return None
        return max(int(self.memory_budget_mb), 1) * 1024 * 1024

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self, progress_callback: ProgressCallback) -> Any:
        self.cancel_token.check_cancelled()

        def emit(message: str, fraction: float) -> None:
            self.cancel_token.check_cancelled()
            progress_callback(message, fraction)

        if self.status_message:
            emit(self.status_message, 0.0)
        result = self.operation(emit)
        self.cancel_token.check_cancelled()
        return result
