from __future__ import annotations

import sys
import time
import unittest
from typing import Any

from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from app.services.computation_task import ComputationCancelled, ComputationTask
from app.widgets.log_panel import LogPanel
from app.widgets.worker_runner import (
    BackgroundWorker,
    WorkerRunner,
    TEXT_ONLY,
)


def _new_app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _RunnerPage(QWidget, WorkerRunner):
    """Minimal QWidget mixing in WorkerRunner, exposing handler calls."""

    def __init__(self, log_panel: LogPanel) -> None:
        super().__init__()
        self.log_panel = log_panel
        self.status_label = QLabel("Idle")
        self._init_worker_runner()
        self.results: list[Any] = []
        self.errors: list[str] = []
        self.cancelled = False
        self.completion_thread: QThread | None = None
        self.progress_lines: list[tuple[str, float]] = []

    def _handle_result(self, result: Any) -> None:
        self.completion_thread = QThread.currentThread()
        self.results.append(result)
        super()._handle_result(result)

    def _handle_error(self, message: str) -> None:
        self.errors.append(message)
        super()._handle_error(message)

    def _handle_progress(self, message: str, fraction: float) -> None:
        self.progress_lines.append((message, fraction))
        super()._handle_progress(message, fraction)

    def _handle_cancelled(self) -> None:
        self.cancelled = True
        super()._handle_cancelled()


class BackgroundWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _new_app()

    def test_invokes_operation_with_progress_callback(self) -> None:
        captured: dict[str, Any] = {}

        def operation(emit):
            captured["called"] = True
            emit("stage", 0.5)
            return 42

        worker = BackgroundWorker(operation)
        results: list[Any] = []
        progresses: list[tuple[str, float]] = []
        worker.finished.connect(results.append)
        worker.progress.connect(lambda m, f: progresses.append((m, f)))
        worker.run()  # synchronous on the current thread
        self.assertTrue(captured["called"])
        self.assertEqual(results, [42])
        self.assertEqual(progresses, [("stage", 0.5)])

    def test_failed_signal_on_exception(self) -> None:
        def operation(_emit):
            raise ValueError("boom")

        worker = BackgroundWorker(operation)
        failures: list[str] = []
        worker.failed.connect(failures.append)
        worker.run()
        self.assertEqual(failures, ["boom"])

    def test_stdout_capture_forwards_text_only_progress(self) -> None:
        def operation(emit):
            print("captured 75%")
            return None

        worker = BackgroundWorker(operation, capture_stdout=True)
        progresses: list[tuple[str, float]] = []
        worker.progress.connect(lambda m, f: progresses.append((m, f)))
        worker.run()
        self.assertTrue(any(msg == "captured 75%" and f is msg or True for msg, f in progresses))
        # The stdout line must arrive with TEXT_ONLY fraction (NaN).
        self.assertTrue(any(f != f for _msg, f in progresses))  # NaN check

    def test_capture_stdout_false_skips_stream(self) -> None:
        captured: list[str] = []

        def operation(emit):
            print("should-not-leak")
            emit("done", 1.0)
            return None

        worker = BackgroundWorker(operation, capture_stdout=False)
        # Replace real stdout so the print doesn't pollute test output.
        class _Spy:
            def write(self, data):
                captured.append(data)

            def flush(self):
                pass

        import io

        original = sys.stdout
        sys.stdout = _Spy()
        try:
            progresses: list[tuple[str, float]] = []
            worker.progress.connect(lambda m, f: progresses.append((m, f)))
            worker.run()
        finally:
            sys.stdout = original
        self.assertIn("should-not-leak", "".join(captured))
        self.assertEqual(progresses, [("done", 1.0)])

    def test_cancel_request_stops_cooperative_operation(self) -> None:
        worker = BackgroundWorker(lambda emit: emit("step", 0.5), capture_stdout=False)
        cancelled: list[bool] = []
        worker.cancelled.connect(lambda: cancelled.append(True))

        worker.cancel()
        worker.run()

        self.assertEqual(cancelled, [True])

    def test_computation_task_runs_and_cancels_through_worker(self) -> None:
        task = ComputationTask(
            "Task",
            lambda emit: (emit("step", 0.25), 7)[1],
            memory_budget_mb=64,
            result_key="sample",
            status_message="Starting task",
        )
        worker = BackgroundWorker(task, capture_stdout=False)
        results: list[int] = []
        progresses: list[tuple[str, float]] = []
        worker.finished.connect(results.append)
        worker.progress.connect(lambda message, fraction: progresses.append((message, fraction)))

        worker.run()

        self.assertEqual(results, [7])
        self.assertEqual(task.memory_budget_bytes, 64 * 1024 * 1024)
        self.assertIn(("Starting task", 0.0), progresses)
        self.assertIn(("step", 0.25), progresses)

        cancelled_task = ComputationTask("Cancelled", lambda _emit: None)
        cancelled_task.cancel()
        with self.assertRaises(ComputationCancelled):
            cancelled_task.run(lambda _message, _fraction: None)


class WorkerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _new_app()

    def _run_until_thread_finishes(self, page: _RunnerPage) -> None:
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        assert page.worker_thread is not None
        page.worker_thread.finished.connect(loop.quit)
        timer.start(3000)
        loop.exec()
        self.app.processEvents()
        self.assertIsNone(
            page.worker_thread, "Worker thread cleanup did not complete before hard timeout."
        )

    def test_init_worker_runner_sets_defaults(self) -> None:
        page = _RunnerPage(LogPanel())
        self.assertIsNone(page.worker_thread)
        self.assertIsNone(page.worker)
        self.assertEqual(page.pending_operation, "")
        self.assertFalse(page._is_busy)
        self.assertFalse(page._is_running())

    def test_start_background_dispatches_result_on_gui_thread(self) -> None:
        page = _RunnerPage(LogPanel())

        def operation(emit):
            emit("step", 0.3)
            return "payload"

        started = page._start_background("Job", operation)
        self.assertTrue(started)
        self.assertTrue(page._is_busy)
        self._run_until_thread_finishes(page)
        self.assertEqual(page.results, ["payload"])
        self.assertIs(page.completion_thread, self.app.thread())
        self.assertFalse(page._is_busy)
        self.assertIsNone(page.worker)
        self.assertIsNone(page.worker_thread)
        self.assertIn(("step", 0.3), page.progress_lines)

    def test_failure_clears_state_and_logs(self) -> None:
        page = _RunnerPage(LogPanel())

        def operation(_emit):
            raise RuntimeError("expected failure")

        page._start_background("Failing Job", operation)
        self._run_until_thread_finishes(page)
        self.assertEqual(page.errors, ["expected failure"])
        self.assertFalse(page._is_busy)
        self.assertIsNone(page.worker)
        self.assertIn("Failed", page.status_label.text())

    def test_reentry_guard_returns_false_without_leaking(self) -> None:
        page = _RunnerPage(LogPanel())

        started_first = []

        def slow_operation(emit):
            # Hold the worker thread busy long enough to attempt re-entry.
            time.sleep(0.4)
            return "first"

        def start_first():
            started_first.append(page._start_background("First", slow_operation))

        start_first()
        self.assertTrue(started_first[0])

        # Second start must be rejected while the first is in flight.
        second = page._start_background("Second", lambda _emit: "second")
        self.assertFalse(second)
        self.assertTrue(page._is_busy)
        # The pending operation is unchanged by the rejected attempt.
        self.assertEqual(page.pending_operation, "First")

        self._run_until_thread_finishes(page)
        self.assertEqual(page.results, ["first"])
        self.assertFalse(page._is_busy)

    def test_cancel_background_requests_cooperative_stop(self) -> None:
        page = _RunnerPage(LogPanel())
        page.worker = BackgroundWorker(lambda _emit: None)

        self.assertTrue(page.cancel_background())
        self.assertTrue(page.worker._cancel_requested.is_set())

    def test_text_only_progress_forwarded_without_fraction(self) -> None:
        page = _RunnerPage(LogPanel())

        def operation(emit):
            print("captured text 80%")
            return None

        page._start_background("Job", operation)
        self._run_until_thread_finishes(page)
        # At least one text-only line must have arrived.
        self.assertTrue(any(f != f for _msg, f in page.progress_lines))

    def test_fraction_progress_forwarded_with_percent(self) -> None:
        page = _RunnerPage(LogPanel())
        seen_values: list[int] = []
        page.log_panel.progress.valueChanged.connect(seen_values.append)

        def operation(emit):
            emit("stage", 0.5)
            return None

        page._start_background("Job", operation)
        self._run_until_thread_finishes(page)
        self.assertIn(("stage", 0.5), page.progress_lines)
        # The 50% must have been observed at some point before completion
        # (the finished handler resets the bar to 100%).
        self.assertIn(50, seen_values)

    def test_finish_sync_runs_on_gui_thread(self) -> None:
        page = _RunnerPage(LogPanel())

        def operation():
            return "sync-payload"

        page._finish_sync("Sync Job", operation)
        self.assertEqual(page.results, ["sync-payload"])
        self.assertIs(page.completion_thread, self.app.thread())
        self.assertEqual(page.pending_operation, "Sync Job")

    def test_finish_sync_routes_exceptions_to_error_handler(self) -> None:
        page = _RunnerPage(LogPanel())

        def operation():
            raise ValueError("sync boom")

        page._finish_sync("Sync Job", operation)
        self.assertEqual(page.errors, ["sync boom"])

    def test_start_background_accepts_computation_task_metadata(self) -> None:
        page = _RunnerPage(LogPanel())
        task = ComputationTask(
            "Task Job",
            lambda emit: (emit("half", 0.5), "done")[1],
            memory_budget_mb=32,
            result_key="task-result",
            status_message="Preparing task",
            parameters={"alpha": 1},
        )

        self.assertTrue(page._start_background("Task Job", task))
        self._run_until_thread_finishes(page)

        self.assertEqual(page.results, ["done"])
        process_log = page.log_panel.process_log.toPlainText()
        self.assertIn("memory_budget_mb=32", process_log)
        self.assertIn("result_key=task-result", process_log)


if __name__ == "__main__":
    unittest.main()
