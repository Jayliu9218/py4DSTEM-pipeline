from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QTabWidget,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ProcessSnapshot:
    step: str
    parameters: dict[str, object]
    warnings: tuple[str, ...] = ()


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.event_log = self._make_output()
        self.process_log = self._make_output()
        self.warning_log = self._make_output()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self._current_progress = 0
        # Single-line status display (latest log line only).
        self.status_line = QLineEdit()
        self.status_line.setReadOnly(True)
        self.status_line.setPlaceholderText("Idle")
        self.tabs = QTabWidget()
        self.tabs.addTab(self._progress_panel(), "Progress")
        self.tabs.addTab(self.event_log, "Activity Log")
        self.tabs.addTab(self.warning_log, "Warnings")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self.tabs)

    def log(self, message: str) -> None:
        self.event_log.appendPlainText(self._timestamped(message))
        if "warn" in message.lower() or "fail" in message.lower() or "error" in message.lower():
            self.warning_log.appendPlainText(self._timestamped(message))

    def process(self, message: str) -> None:
        self.process_log.appendPlainText(self._timestamped(message))

    def process_started(self, name: str, details: str = "") -> None:
        self._current_progress = 0
        self.progress.setRange(0, 0)
        suffix = f" | {details}" if details else ""
        self.status_line.setText(f"START {name}{suffix}")
        self.process(f"START {name}{suffix}")

    def process_finished(self, name: str, details: str = "") -> None:
        self._current_progress = 100
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        suffix = f" | {details}" if details else ""
        self.status_line.setText(f"DONE  {name}{suffix}")
        self.process(f"DONE  {name}{suffix}")

    def process_failed(self, name: str, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(self._current_progress)
        self.status_line.setText(f"FAIL  {name} | {message}")
        self.process(f"FAIL  {name} | {message}")
        self.warning_log.appendPlainText(self._timestamped(f"FAIL  {name} | {message}"))

    def process_progress(self, message: str) -> None:
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
        if match:
            value = min(max(float(match.group(1)), 0), 100)
            self._current_progress = max(self._current_progress, int(value))
            self.progress.setRange(0, 100)
            self.progress.setValue(self._current_progress)
        else:
            self.progress.setRange(0, 0)
        self.status_line.setText(message)
        self.process(f"PROGRESS {message}")

    def process_snapshot(self, snapshot: ProcessSnapshot) -> None:
        self.process(f"STEP  {snapshot.step}")
        if snapshot.parameters:
            params = ", ".join(f"{key}={value}" for key, value in snapshot.parameters.items())
            self.process(f"PARAM {params}")
        for warning in snapshot.warnings:
            self.process(f"WARN  {warning}")

    def _make_output(self) -> QPlainTextEdit:
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setMaximumBlockCount(2000)
        return output

    def _progress_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        # Compact: progress bar (no text) + single-line status.
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(4)
        progress_row.addWidget(self.progress, 1)
        layout.addLayout(progress_row)
        layout.addWidget(self.status_line)
        return panel

    def _timestamped(self, message: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
