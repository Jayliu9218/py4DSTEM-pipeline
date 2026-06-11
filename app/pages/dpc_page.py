from __future__ import annotations

import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.phase_contrast_service import (
    DPCParams,
    PhaseContrastResult,
    PhaseContrastService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.progress_stream import ProgressStream


class DPCWorker(QObject):
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


class DPCPage(QWidget):
    dpc_result_ready = Signal(object)

    DIAGNOSTIC_KEYS = [
        "Complex CoM",
        "CoM X",
        "CoM Y",
        "Phase",
        "Measured CoM X",
        "Measured CoM Y",
        "Normalized CoM X",
        "Normalized CoM Y",
        "Potential",
    ]

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = PhaseContrastService()
        self.worker_thread: QThread | None = None
        self.worker: DPCWorker | None = None
        self.current_process_step = WorkflowStep.DPC
        self.result: PhaseContrastResult | None = None

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        self.dpc_energy = self._float_input(10, 1000000, 200000, decimals=0, unit="eV")
        self.dpc_force_rotation = self._float_input(-360, 360, 0, decimals=2, unit="deg")
        self.dpc_force_transpose = QComboBox()
        self.dpc_force_transpose.addItems(["Off", "On"])

        self.dpc_use_segmented = QComboBox()
        self.dpc_use_segmented.addItems(["Full pixelated COM", "Virtual segmented detector"])
        self.dpc_mask_mode = QComboBox()
        self.dpc_mask_mode.addItems(["BF disk / annular mask", "Annular only"])
        self.dpc_inner_angle = self._float_input(0, 1000, 0, decimals=1, unit="mrad")
        self.dpc_outer_angle = self._float_input(0, 1000, 25, decimals=1, unit="mrad")

        self.dpc_origin_x = self._float_input(-10000, 10000, 0, decimals=2, unit="px")
        self.dpc_origin_y = self._float_input(-10000, 10000, 0, decimals=2, unit="px")
        self.dpc_descan = QComboBox()
        self.dpc_descan.addItems(["On", "Off"])
        self.dpc_background = QComboBox()
        self.dpc_background.addItems(["None", "Subtract mean", "Subtract median"])

        self.dpc_padding = self._int_input(0, 512, 0, unit="px")
        self.dpc_regularization = QComboBox()
        self.dpc_regularization.addItems(["None", "Gaussian", "Butterworth"])
        self.dpc_gaussian_sigma = self._float_input(0, 1000, 0, decimals=2, unit="A")
        self.dpc_q_lowpass = self._float_input(0, 1000, 0, decimals=3, unit="A^-1")
        self.dpc_q_highpass = self._float_input(0, 1000, 0, decimals=3, unit="A^-1")

        self.image_selector = QComboBox()
        self.image_selector.addItems(self.DIAGNOSTIC_KEYS)
        self.image_selector.currentTextChanged.connect(self._show_selected_image)

        self.run_button = QPushButton("Reconstruct")
        self.run_button.clicked.connect(self._run)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(0, 0)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.rotation_label = QLabel("")
        self.rotation_label.setWordWrap(True)

        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def _build_layout(self) -> None:
        input_group = QGroupBox("1. Input")
        input_form = QFormLayout(input_group)
        input_form.addRow("Energy", self.dpc_energy)

        detector_group = QGroupBox("2. Detector / Mask")
        detector_form = QFormLayout(detector_group)
        detector_form.addRow("Detector mode", self.dpc_use_segmented)
        detector_form.addRow("Mask mode", self.dpc_mask_mode)
        detector_form.addRow("Inner angle", self.dpc_inner_angle)
        detector_form.addRow("Outer angle", self.dpc_outer_angle)

        calibration_group = QGroupBox("3. Calibration")
        calibration_form = QFormLayout(calibration_group)
        calibration_form.addRow("Origin x", self.dpc_origin_x)
        calibration_form.addRow("Origin y", self.dpc_origin_y)
        calibration_form.addRow("Force rotation", self.dpc_force_rotation)
        calibration_form.addRow("Force transpose", self.dpc_force_transpose)
        calibration_form.addRow("Descan", self.dpc_descan)
        calibration_form.addRow("Background", self.dpc_background)

        recon_group = QGroupBox("4. Reconstruction")
        recon_form = QFormLayout(recon_group)
        recon_form.addRow("Padding", self.dpc_padding)
        recon_form.addRow("Regularization", self.dpc_regularization)
        recon_form.addRow("Gaussian sigma", self.dpc_gaussian_sigma)
        recon_form.addRow("Q lowpass", self.dpc_q_lowpass)
        recon_form.addRow("Q highpass", self.dpc_q_highpass)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(input_group)
        layout.addWidget(detector_group)
        layout.addWidget(calibration_group)
        layout.addWidget(recon_group)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("View result:"))
        layout.addWidget(self.image_selector)
        layout.addWidget(self.rotation_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.controls_panel = controls

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.viewer, 1)

    def _show_selected_image(self, key: str) -> None:
        if self.result is None or key not in self.result.images:
            return
        image = self.result.images[key]
        if image is not None:
            self.viewer.set_image(np.asarray(image))

    def _current_params(self) -> DPCParams:
        force_rotation = self.dpc_force_rotation.value()
        return DPCParams(
            energy=self.dpc_energy.value(),
            plot_center_of_mass=self.dpc_use_segmented.currentText().lower(),
            force_com_rotation=force_rotation if force_rotation != 0 else None,
            force_com_transpose=self.dpc_force_transpose.currentText() == "On",
        )

    def _run(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "DPC", "Load a 4D DataCube first.")
            return

        if not self.workflow_state.is_completed(WorkflowStep.BF_DF_PREVIEW):
            result = QMessageBox.question(
                self, "DPC",
                "Recommended: run BF/DF Preview first to check the bright-field disk.\nContinue anyway?",
            )
            if result != QMessageBox.Yes:
                return

        params = self._current_params()
        self.status_label.setText("Running DPC...")
        self.run_button.setEnabled(False)
        self.log_panel.log("DPC reconstruction started")
        self.log_panel.process_started("DPC", "DPC")
        self.log_panel.process_snapshot(
            ProcessSnapshot(step="DPC", parameters={"method": "DPC", "energy": params.energy}),
        )

        operation = lambda: self.service.run_dpc(source, params)

        self.worker_thread = QThread()
        self.worker = DPCWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.progress.connect(self._handle_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _handle_finished(self, result: PhaseContrastResult) -> None:
        self.result = result
        self.run_button.setEnabled(True)
        elapsed_str = f"{result.elapsed_seconds:.1f}s"
        self.status_label.setText(f"Done in {elapsed_str}")
        self.log_panel.process_finished("DPC done", elapsed_str)

        if result.rotation_degrees is not None:
            self.rotation_label.setText(f"Estimated rotation: {result.rotation_degrees:.1f}\u00b0")

        available_keys = [k for k in self.DIAGNOSTIC_KEYS if k in result.images and result.images[k] is not None]
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        self.image_selector.addItems(available_keys)
        self.image_selector.blockSignals(False)

        default_key = "Complex CoM" if "Complex CoM" in available_keys else (available_keys[0] if available_keys else None)
        if default_key is not None:
            self.image_selector.setCurrentText(default_key)
            image = result.images[default_key]
            if image is not None:
                self.viewer.set_image(np.asarray(image))

        if self.result_registry is not None:
            for name, image in result.images.items():
                key = f"dpc_{name.lower().replace(' ', '_')}"
                self.result_registry.register(
                    name=key, category="Phase Retrieval", data=image,
                    export_formats=("npy", "png", "tiff"),
                )

        self.dpc_result_ready.emit(result)
        self.workflow_state.mark_completed(self.current_process_step)

    def _handle_failed(self, error: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"DPC reconstruction failed: {error}")
        self.log_panel.process_finished("DPC failed", error)

    def _handle_progress(self, message: str) -> None:
        self.log_panel.log(message)
        m = re.search(r"(\d+)%", message)
        if m:
            pct = int(m.group(1))
            self.log_panel.process_progress(f"DPC {pct}%")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.DPC]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _watch_parameters(self) -> None:
        for spin in [self.dpc_energy, self.dpc_force_rotation]:
            self.workflow_state.watch(spin, WorkflowStep.DPC, "valueChanged")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(self, minimum, maximum, value, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "dpc_energy": self.dpc_energy.value(),
            "dpc_force_rotation": self.dpc_force_rotation.value(),
            "dpc_force_transpose": self.dpc_force_transpose.currentText(),
            "dpc_use_segmented": self.dpc_use_segmented.currentText(),
            "dpc_mask_mode": self.dpc_mask_mode.currentText(),
            "dpc_inner_angle": self.dpc_inner_angle.value(),
            "dpc_outer_angle": self.dpc_outer_angle.value(),
            "dpc_origin_x": self.dpc_origin_x.value(),
            "dpc_origin_y": self.dpc_origin_y.value(),
            "dpc_descan": self.dpc_descan.currentText(),
            "dpc_background": self.dpc_background.currentText(),
            "dpc_padding": int(self.dpc_padding.value()),
            "dpc_regularization": self.dpc_regularization.currentText(),
            "dpc_gaussian_sigma": self.dpc_gaussian_sigma.value(),
            "dpc_q_lowpass": self.dpc_q_lowpass.value(),
            "dpc_q_highpass": self.dpc_q_highpass.value(),
        }