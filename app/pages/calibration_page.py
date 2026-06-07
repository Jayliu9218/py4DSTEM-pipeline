from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import BraggStrainService, CalibrationActionResult
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel
from app.widgets.progress_stream import ProgressStream


class CalibrationWorker(QObject):
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


class CalibrationPage(QWidget):
    def __init__(
        self,
        datacube_provider: Callable[[], object | None],
        braggvectors_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
        workflow_state: WorkflowState,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.braggvectors_provider = braggvectors_provider
        self.service = service
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.worker_thread: QThread | None = None
        self.worker: CalibrationWorker | None = None
        self.current_process_name = "Calibration step"
        self.current_process_step = WorkflowStep.CALIBRATION_APPLY

        self.source_label = QLabel("-")
        self.origin_label = QLabel("-")
        self.ellipse_label = QLabel("-")
        self.pixel_label = QLabel("-")
        self.rotate_label = QLabel("-")
        self.complete_label = QLabel("-")
        self.applied_label = QLabel("none")
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
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
        self.center_check = QCheckBox("Origin")
        self.ellipse_check = QCheckBox("Ellipse")
        self.pixel_check = QCheckBox("Pixel size")
        self.rotate_check = QCheckBox("QR rotation")
        self.apply_button = QPushButton("Apply Selected Corrections")
        self.buttons = [
            self.refresh_button,
            self.origin_button,
            self.ellipse_button,
            self.pixel_button,
            self.rotation_button,
            self.apply_button,
        ]
        self.viewers = QTabWidget()

        self.refresh_button.clicked.connect(self.refresh_status)
        self.origin_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.calibrate_origin(self.braggvectors_provider()),
                "Measure and fit origin",
                WorkflowStep.CALIBRATION_ORIGIN,
            )
        )
        self.ellipse_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.calibrate_ellipse(
                    self.braggvectors_provider(),
                    self.ellipse_inner.value(),
                    self.ellipse_outer.value(),
                    self.sampling_spin.value(),
                ),
                "Fit ellipticity",
                WorkflowStep.CALIBRATION_ELLIPSE,
            )
        )
        self.pixel_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_pixel_size(
                    self.braggvectors_provider(), self.pixel_spin.value()
                ),
                "Set Q pixel size",
                WorkflowStep.CALIBRATION_PIXEL,
            )
        )
        self.rotation_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_qr_rotation(
                    self.braggvectors_provider(), self.rotation_spin.value()
                ),
                "Set QR rotation",
                WorkflowStep.CALIBRATION_ROTATION,
            )
        )
        self.apply_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_calibration_state(
                    self.braggvectors_provider(),
                    self.center_check.isChecked(),
                    self.ellipse_check.isChecked(),
                    self.pixel_check.isChecked(),
                    self.rotate_check.isChecked(),
                ),
                "Apply calibration corrections",
                WorkflowStep.CALIBRATION_APPLY,
            )
        )
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def _build_layout(self) -> None:
        status_form = QFormLayout()
        for label, widget in [
            ("Source", self.source_label),
            ("origin", self.origin_label),
            ("ellipse", self.ellipse_label),
            ("pixel", self.pixel_label),
            ("rotate", self.rotate_label),
            ("measurements complete", self.complete_label),
            ("applied corrections", self.applied_label),
        ]:
            status_form.addRow(label, widget)
        controls = QFormLayout()
        controls.addRow("ellipse inner radius", self.ellipse_inner)
        controls.addRow("ellipse outer radius", self.ellipse_outer)
        controls.addRow("BVM sampling", self.sampling_spin)
        controls.addRow("Q pixel size (A^-1)", self.pixel_spin)
        controls.addRow("QR rotation (degrees)", self.rotation_spin)
        correction_row = QHBoxLayout()
        for checkbox in [
            self.center_check,
            self.ellipse_check,
            self.pixel_check,
            self.rotate_check,
        ]:
            correction_row.addWidget(checkbox)
        controls.addRow("Manual correction state", correction_row)
        buttons = QVBoxLayout()
        for button in self.buttons:
            buttons.addWidget(button)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(status_form)
        left_layout.addLayout(controls)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        left.setFixedWidth(430)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viewers)
        splitter.setSizes([430, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

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
        if braggvectors is not None:
            state = braggvectors.calstate
            for checkbox, name in [
                (self.center_check, "center"),
                (self.ellipse_check, "ellipse"),
                (self.pixel_check, "pixel"),
                (self.rotate_check, "rotate"),
            ]:
                checkbox.blockSignals(True)
                checkbox.setChecked(state.get(name, False))
                checkbox.blockSignals(False)
            enabled = [name for name, value in state.items() if value]
            self.applied_label.setText(", ".join(enabled) if enabled else "none")
        else:
            self.applied_label.setText("none")

    def _run(
        self,
        operation,
        process_name: str = "Calibration step",
        process_step: str = WorkflowStep.CALIBRATION_APPLY,
    ) -> None:
        if self.braggvectors_provider() is None:
            QMessageBox.information(self, "Calibration", "Run full BraggVectors first.")
            return
        for button in self.buttons:
            button.setEnabled(False)
        self.status_label.setText("Running calibration step...")
        self.current_process_name = process_name
        self.current_process_step = process_step
        self.log_panel.process_started(process_name)
        self.worker_thread = QThread()
        self.worker = CalibrationWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.progress.connect(self.log_panel.process_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _finished(self, result: CalibrationActionResult) -> None:
        self.status_label.setText(f"{result.message} ({result.elapsed_seconds:.2f} s)")
        self.log_panel.log(result.message)
        self.log_panel.process_finished(self.current_process_name, result.message)
        self.viewers.clear()
        for name, image in result.images.items():
            viewer = ImageViewer()
            viewer.set_image(image)
            self.viewers.addTab(viewer, name)
        self.refresh_status()
        self.workflow_state.mark_completed(self.current_process_step)

    def _failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Calibration failed: {message}")
        self.log_panel.process_failed(self.current_process_name, message)
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

    def _watch_parameters(self) -> None:
        for spin in [self.ellipse_inner, self.ellipse_outer, self.sampling_spin]:
            self.workflow_state.watch(spin, WorkflowStep.CALIBRATION_ELLIPSE, "valueChanged")
        self.workflow_state.watch(
            self.pixel_spin, WorkflowStep.CALIBRATION_PIXEL, "valueChanged"
        )
        self.workflow_state.watch(
            self.rotation_spin, WorkflowStep.CALIBRATION_ROTATION, "valueChanged"
        )
        for checkbox in [
            self.center_check,
            self.ellipse_check,
            self.pixel_check,
            self.rotate_check,
        ]:
            self.workflow_state.watch(checkbox, WorkflowStep.CALIBRATION_APPLY, "toggled")

    def _refresh_stale_status(self) -> None:
        steps = [
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_ELLIPSE,
            WorkflowStep.CALIBRATION_PIXEL,
            WorkflowStep.CALIBRATION_ROTATION,
            WorkflowStep.CALIBRATION_APPLY,
        ]
        if self.workflow_state.any_stale(steps):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
