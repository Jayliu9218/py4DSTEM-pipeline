from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.services.orientation_service import OrientationPlanParams, OrientationResult, OrientationService
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel
from app.widgets.progress_stream import ProgressStream


class OrientationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            stream = ProgressStream(self.progress.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                self.finished.emit(self.operation())
            stream.flush()
        except Exception as exc:
            self.failed.emit(str(exc))


class OrientationPage(QWidget):
    def __init__(self, braggvectors_provider: Callable[[], object | None], log_panel: LogPanel) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.log_panel = log_panel
        self.service = OrientationService()
        self.worker_thread: QThread | None = None
        self.worker: OrientationWorker | None = None
        self.crystal_label = QLabel("No CIF loaded")
        self.status_label = QLabel("Idle")
        self.voltage = self._spin(1000, 1000000, 300000)
        self.k_max = self._spin(0.1, 10, 1.5)
        self.zone_step = self._spin(0.1, 30, 2)
        self.plane_step = self._spin(0.1, 30, 2)
        self.load_button = QPushButton("7.1 Load Crystal CIF")
        self.plan_button = QPushButton("7.2 Create Orientation Plan")
        self.match_button = QPushButton("7.3 Match and Show Orientation Map")
        self.buttons = [self.load_button, self.plan_button, self.match_button]
        self.viewer = ImageViewer()
        self.load_button.clicked.connect(self.load_cif)
        self.plan_button.clicked.connect(lambda: self._run(lambda: self.service.create_plan(self._params())))
        self.match_button.clicked.connect(
            lambda: self._run(lambda: self.service.match(self.braggvectors_provider()))
        )
        form = QFormLayout()
        form.addRow("Crystal", self.crystal_label)
        form.addRow("accelerating voltage", self.voltage)
        form.addRow("k_max", self.k_max)
        form.addRow("zone-axis angle step", self.zone_step)
        form.addRow("in-plane angle step", self.plane_step)
        row = QVBoxLayout()
        for button in self.buttons:
            row.addWidget(button)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(form)
        left_layout.addLayout(row)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        left.setFixedWidth(430)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viewer)
        splitter.setSizes([430, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def load_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load crystal structure", "", "CIF files (*.cif)")
        if not path:
            return
        try:
            self.crystal_label.setText(self.service.load_crystal(path))
            self.status_label.setText("Crystal loaded")
        except Exception as exc:
            QMessageBox.warning(self, "Orientation", str(exc))

    def _params(self) -> OrientationPlanParams:
        return OrientationPlanParams(
            accelerating_voltage=self.voltage.value(),
            k_max=self.k_max.value(),
            angle_step_zone_axis=self.zone_step.value(),
            angle_step_in_plane=self.plane_step.value(),
        )

    def _run(self, operation) -> None:
        for button in self.buttons:
            button.setEnabled(False)
        self.status_label.setText("Running orientation step...")
        self.log_panel.process_started("Orientation analysis")
        self.worker_thread = QThread()
        self.worker = OrientationWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.progress.connect(self.log_panel.process_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear)
        self.worker_thread.start()

    def _finished(self, result) -> None:
        if isinstance(result, OrientationResult):
            self.viewer.set_image(result.preview.mean(axis=2))
            self.status_label.setText(f"Orientation map ready in {result.elapsed_seconds:.2f} s")
        else:
            self.status_label.setText(f"Orientation plan ready in {float(result):.2f} s")
        self.log_panel.log(self.status_label.text())
        self.log_panel.process_finished("Orientation analysis", self.status_label.text())

    def _failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Orientation failed: {message}")
        self.log_panel.process_failed("Orientation analysis", message)
        QMessageBox.warning(self, "Orientation", message)

    def _clear(self) -> None:
        self.worker = None
        self.worker_thread = None
        for button in self.buttons:
            button.setEnabled(True)

    def _spin(self, minimum, maximum, value) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin
