from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import BraggStrainService, CalibrationActionResult
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class CalibrationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.finished.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))


class CalibrationPage(QWidget):
    def __init__(
        self,
        datacube_provider: Callable[[], object | None],
        braggvectors_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.braggvectors_provider = braggvectors_provider
        self.service = service
        self.log_panel = log_panel
        self.worker_thread: QThread | None = None
        self.worker: CalibrationWorker | None = None

        self.source_label = QLabel("-")
        self.origin_label = QLabel("-")
        self.ellipse_label = QLabel("-")
        self.pixel_label = QLabel("-")
        self.rotate_label = QLabel("-")
        self.complete_label = QLabel("-")
        self.status_label = QLabel("Idle")
        self.ellipse_inner = self._float_spin(0.1, 100000, 290)
        self.ellipse_outer = self._float_spin(0.1, 100000, 360)
        self.sampling_spin = QSpinBox()
        self.sampling_spin.setRange(1, 64)
        self.sampling_spin.setValue(8)
        self.pixel_spin = self._float_spin(0.000001, 1000, 0.02, decimals=6)
        self.rotation_spin = self._float_spin(-360, 360, -83)
        self.refresh_button = QPushButton("Refresh Status")
        self.origin_button = QPushButton("6.1 Measure/Fit Origin")
        self.ellipse_button = QPushButton("6.2 Fit Ellipticity")
        self.pixel_button = QPushButton("6.3 Set Pixel Size")
        self.rotation_button = QPushButton("6.4 Set QR Rotation")
        self.buttons = [
            self.refresh_button,
            self.origin_button,
            self.ellipse_button,
            self.pixel_button,
            self.rotation_button,
        ]
        self.viewers = QTabWidget()

        self.refresh_button.clicked.connect(self.refresh_status)
        self.origin_button.clicked.connect(
            lambda: self._run(lambda: self.service.calibrate_origin(self.braggvectors_provider()))
        )
        self.ellipse_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.calibrate_ellipse(
                    self.braggvectors_provider(),
                    self.ellipse_inner.value(),
                    self.ellipse_outer.value(),
                    self.sampling_spin.value(),
                )
            )
        )
        self.pixel_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_pixel_size(
                    self.braggvectors_provider(), self.pixel_spin.value()
                )
            )
        )
        self.rotation_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_qr_rotation(
                    self.braggvectors_provider(), self.rotation_spin.value()
                )
            )
        )
        self._build_layout()

    def _build_layout(self) -> None:
        status_form = QFormLayout()
        for label, widget in [
            ("Source", self.source_label),
            ("origin", self.origin_label),
            ("ellipse", self.ellipse_label),
            ("pixel", self.pixel_label),
            ("rotate", self.rotate_label),
            ("complete", self.complete_label),
        ]:
            status_form.addRow(label, widget)
        controls = QFormLayout()
        controls.addRow("ellipse inner radius", self.ellipse_inner)
        controls.addRow("ellipse outer radius", self.ellipse_outer)
        controls.addRow("BVM sampling", self.sampling_spin)
        controls.addRow("Q pixel size (A^-1)", self.pixel_spin)
        controls.addRow("QR rotation (degrees)", self.rotation_spin)
        buttons = QHBoxLayout()
        for button in self.buttons:
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(status_form)
        layout.addLayout(controls)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        layout.addWidget(self.viewers, 1)

    def refresh_status(self) -> None:
        braggvectors = self.braggvectors_provider()
        source = braggvectors if braggvectors is not None else self.datacube_provider()
        status = self.service.calibration_status(source)
        self.source_label.setText("BraggVectors" if braggvectors is not None else "DataCube")
        self.origin_label.setText(status.origin)
        self.ellipse_label.setText(status.ellipse)
        self.pixel_label.setText(status.pixel)
        self.rotate_label.setText(status.rotate)
        self.complete_label.setText("yes" if status.complete else "no")

    def _run(self, operation) -> None:
        if self.braggvectors_provider() is None:
            QMessageBox.information(self, "Calibration", "Run full BraggVectors first.")
            return
        for button in self.buttons:
            button.setEnabled(False)
        self.status_label.setText("Running calibration step...")
        self.worker_thread = QThread()
        self.worker = CalibrationWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _finished(self, result: CalibrationActionResult) -> None:
        self.status_label.setText(f"{result.message} ({result.elapsed_seconds:.2f} s)")
        self.log_panel.log(result.message)
        self.viewers.clear()
        for name, image in result.images.items():
            viewer = ImageViewer()
            viewer.set_image(image)
            self.viewers.addTab(viewer, name)
        self.refresh_status()

    def _failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Calibration failed: {message}")
        QMessageBox.warning(self, "Calibration", message)

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        for button in self.buttons:
            button.setEnabled(True)

    def _float_spin(self, minimum, maximum, value, decimals=2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin
