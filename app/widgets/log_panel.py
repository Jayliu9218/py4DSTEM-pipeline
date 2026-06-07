from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QSplitter, QVBoxLayout, QWidget


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.event_log = self._make_output()
        self.process_log = self._make_output()

        event_panel = self._labeled_panel("Activity Log", self.event_log)
        process_panel = self._labeled_panel("Calculation Process", self.process_log)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(event_panel)
        splitter.addWidget(process_panel)
        splitter.setSizes([600, 600])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def log(self, message: str) -> None:
        self.event_log.appendPlainText(self._timestamped(message))

    def process(self, message: str) -> None:
        self.process_log.appendPlainText(self._timestamped(message))

    def process_started(self, name: str, details: str = "") -> None:
        suffix = f" | {details}" if details else ""
        self.process(f"START {name}{suffix}")

    def process_finished(self, name: str, details: str = "") -> None:
        suffix = f" | {details}" if details else ""
        self.process(f"DONE  {name}{suffix}")

    def process_failed(self, name: str, message: str) -> None:
        self.process(f"FAIL  {name} | {message}")

    def process_progress(self, message: str) -> None:
        self.process(f"PROGRESS {message}")

    def _make_output(self) -> QPlainTextEdit:
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setMaximumBlockCount(2000)
        return output

    def _labeled_panel(self, title: str, output: QPlainTextEdit) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(output)
        return panel

    def _timestamped(self, message: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
