from __future__ import annotations

import math
import threading
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from app.widgets.progress_stream import ProgressStream

ProgressCallback = Callable[[str, float], None]
Operation = Callable[[ProgressCallback], Any]

# Sentinel fraction for stdout-captured text that carries no numeric fraction.
# Workers forward such lines with this value; receivers treat NaN as "text only".
TEXT_ONLY = float("nan")


class OperationCancelled(RuntimeError):
    """Raised when a cooperative background operation is cancelled."""


class BackgroundWorker(QObject):
    """Single worker moved to a QThread; runs an operation off the GUI thread.

    Replaces the 13 page-local ``XxxWorker`` classes. The ``operation`` callable
    receives a ``ProgressCallback`` so services can push ``(message, fraction)``
    updates directly. When ``capture_stdout`` is enabled (default), py4DSTEM/tqdm
    output is captured via :class:`ProgressStream` and forwarded as text-only
    progress lines (``fraction=TEXT_ONLY``), so hot internal loops that never
    call the callback still produce visible progress.
    """

    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(str, float)

    def __init__(self, operation: Operation, capture_stdout: bool = True) -> None:
        super().__init__()
        self._operation = operation
        self._capture_stdout = capture_stdout
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    def _emit_progress(self, message: str, fraction: float) -> None:
        if self._cancel_requested.is_set():
            raise OperationCancelled("Operation cancelled.")
        self.progress.emit(message, fraction)

    @Slot()
    def run(self) -> None:
        try:
            if self._capture_stdout:
                stream = ProgressStream(self._stdout_progress)
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = self._operation(self._emit_progress)
                stream.flush()
            else:
                result = self._operation(self._emit_progress)
            if self._cancel_requested.is_set():
                raise OperationCancelled("Operation cancelled.")
            self.finished.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surface any worker failure to the UI
            self.failed.emit(str(exc))

    def _stdout_progress(self, message: str) -> None:
        self._emit_progress(message, TEXT_ONLY)


class WorkerRunner:
    """Mixin providing QThread lifecycle for pages that start background work.

    A page mixes this in and calls :meth:`_start_background` to run an
    :class:`Operation` off the GUI thread. The mixin owns re-entrancy guarding,
    worker/thread construction, signal wiring, and cleanup (``deleteLater`` +
    reference clearing), standardizing on the Ptychography-page teardown model.

    Required page attributes: ``self.log_panel`` (a :class:`~app.widgets.log_panel.LogPanel`).
    Optional: ``self.status_label`` (a ``QLabel``); used by default handlers.

    Pages override :meth:`_handle_result` to dispatch on ``self.pending_operation``.
    """

    def _init_worker_runner(self) -> None:
        """Initialize runner state. Call once from the page ``__init__``."""
        self.worker_thread: QThread | None = None
        self.worker: BackgroundWorker | None = None
        self.pending_operation: str = ""
        self._is_busy: bool = False

    def _is_running(self) -> bool:
        return self._is_busy or self.worker_thread is not None

    def cancel_background(self) -> bool:
        if self.worker is None:
            return False
        self.worker.cancel()
        self.log_panel.process_progress(f"Cancelling {self.pending_operation}...")
        return True

    def _start_background(
        self,
        name: str,
        operation: Operation,
        *,
        capture_stdout: bool = True,
        parameters: dict[str, object] | None = None,
    ) -> bool:
        """Run ``operation`` on a fresh QThread.

        Returns ``True`` if started, ``False`` if a run is already in flight
        (re-entrancy guard). On ``False`` the caller is expected to surface a
        "busy" message to the user; the default behavior logs to ``status_label``.
        """
        if self._is_running():
            self._notify_busy(name)
            return False
        self.pending_operation = name
        self._is_busy = True
        self._on_start(name)
        self.log_panel.process_started(name, name)
        snapshot = self._process_snapshot(name, parameters)
        if snapshot is not None:
            self.log_panel.process_snapshot(snapshot)

        thread = QThread(self)
        worker = BackgroundWorker(operation, capture_stdout=capture_stdout)
        self.worker_thread, self.worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_result)
        worker.failed.connect(self._handle_error)
        worker.cancelled.connect(self._handle_cancelled)
        worker.progress.connect(self._handle_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_refs, Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return True

    def _finish_sync(self, name: str, operation: Callable[[], Any]) -> None:
        """Run ``operation`` synchronously on the GUI thread and dispatch the result.

        Used for cheap service calls (e.g. Ptychography data inspection, geometry,
        QC) that don't warrant a worker thread. Mirrors the async dispatch path so
        ``_handle_result``/``_handle_error`` are reached identically.
        """
        self.pending_operation = name
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001
            self._handle_error(str(exc))
            return
        self._handle_result(result)

    # -- default handlers; pages override as needed ------------------------

    def _on_start(self, name: str) -> None:
        status_label = getattr(self, "status_label", None)
        if status_label is not None:
            status_label.setText(f"Running {name}...")

    def _process_snapshot(self, name: str, parameters: dict[str, object] | None):
        # Local import avoids a hard dependency for pages that don't use snapshots.
        from app.widgets.log_panel import ProcessSnapshot

        params = parameters if parameters is not None else self._default_parameters()
        if not params:
            return None
        return ProcessSnapshot(step=name, parameters=dict(params))

    def _default_parameters(self) -> dict[str, object]:
        snapshot = getattr(self, "params_snapshot", None)
        if callable(snapshot):
            try:
                value = snapshot()
            except Exception:  # noqa: BLE001 - best-effort logging
                return {}
            if isinstance(value, dict):
                return value
        return {}

    def _notify_busy(self, name: str) -> None:
        status_label = getattr(self, "status_label", None)
        if status_label is not None:
            status_label.setText(f"A {name} operation is already running.")
        self.log_panel.log(f"{name}: an operation is already running.")

    def _handle_result(self, result: Any) -> None:
        """Override in pages to dispatch results and mark workflow steps."""
        name = self.pending_operation
        self.log_panel.process_finished(name)
        status_label = getattr(self, "status_label", None)
        if status_label is not None:
            status_label.setText(f"{name} complete.")

    def _handle_error(self, message: str) -> None:
        name = self.pending_operation
        status_label = getattr(self, "status_label", None)
        if status_label is not None:
            status_label.setText(f"Failed: {name}")
        self.log_panel.log(f"{name} failed: {message}")
        self.log_panel.process_failed(name, message)

    def _handle_cancelled(self) -> None:
        name = self.pending_operation
        status_label = getattr(self, "status_label", None)
        if status_label is not None:
            status_label.setText(f"Cancelled: {name}")
        self.log_panel.log(f"{name} cancelled.")
        self.log_panel.process(f"CANCELLED {name}")

    def _handle_progress(self, message: str, fraction: float) -> None:
        if math.isnan(fraction):
            self.log_panel.process_progress(message)
        else:
            pct = max(0, min(100, int(fraction * 100)))
            self.log_panel.process_progress(f"{message} {pct}%")

    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._is_busy = False
