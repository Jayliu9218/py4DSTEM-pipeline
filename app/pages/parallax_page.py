from __future__ import annotations

import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    ParallaxParams,
    PhaseContrastResult,
    PhaseContrastService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.progress_stream import ProgressStream


class ParallaxWorker(QObject):
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


class ParallaxPage(QWidget):
    parallax_result_ready = Signal(object)

    DIAGNOSTIC_KEYS = [
        "Aligned BF",
        "Shift X",
        "Shift Y",
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
        self.worker: ParallaxWorker | None = None
        self.current_process_step = WorkflowStep.PARALLAX
        self.result: PhaseContrastResult | None = None

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        self.energy = self._float_input(10, 1000000, 300000, decimals=0, unit="eV")
        self.padding = self._int_input(0, 128, 16, unit="px")
        self.edge_blend = self._int_input(0, 64, 8, unit="px")
        self.normalize = QCheckBox("Normalize images")
        self.threshold = self._float_input(0, 1, 0.6, decimals=2, unit="")
        self.alignment_bins = QLabel("32,32,32,32,32,32,16,16,16,16,8,8")

        self.image_selector = QComboBox()
        self.image_selector.addItems(self.DIAGNOSTIC_KEYS)
        self.image_selector.currentTextChanged.connect(self._show_selected_image)

        self.rotation_label = QLabel("")
        self.rotation_label.setWordWrap(True)

        self.run_button = QPushButton("Reconstruct")
        self.run_button.clicked.connect(self._run)

        self.workspace = AdaptiveImageWorkspace()

        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def _build_layout(self) -> None:
        input_group = QGroupBox("1. Input")
        input_form = QFormLayout(input_group)
        input_form.addRow("Energy", self.energy)

        processing_group = QGroupBox("2. Processing")
        processing_form = QFormLayout(processing_group)
        processing_form.addRow("Padding", self.padding)
        processing_form.addRow("Edge blend", self.edge_blend)
        processing_form.addRow(self.normalize)
        processing_form.addRow("Threshold intensity", self.threshold)
        processing_form.addRow("Alignment bins", self.alignment_bins)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(input_group)
        layout.addWidget(processing_group)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("View result:"))
        layout.addWidget(self.image_selector)
        layout.addWidget(self.rotation_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.controls_panel = controls

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.workspace, 1)

    def _show_selected_image(self, key: str) -> None:
        if self.result is None or key not in self.result.images:
            return
        image = self.result.images[key]
        if image is not None:
            self.workspace.update_result("selected-preview", FigureResult(f"Selected: {key}", np.asarray(image)))

    def _current_params(self) -> ParallaxParams:
        return ParallaxParams(
            energy=self.energy.value(),
            object_padding_px=(int(self.padding.value()), int(self.padding.value())),
            edge_blend=int(self.edge_blend.value()),
            normalize_images=self.normalize.isChecked(),
            threshold_intensity=self.threshold.value(),
        )

    def _run(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Parallax", "Load a 4D DataCube first.")
            return

        if not self.workflow_state.is_completed(WorkflowStep.DPC):
            result = QMessageBox.question(
                self, "Parallax",
                "Recommended: run DPC first to estimate detector rotation.\nContinue with manual/default parameters?",
            )
            if result != QMessageBox.Yes:
                return

        params = self._current_params()
        self.status_label.setText("Running Parallax...")
        self.run_button.setEnabled(False)
        self.log_panel.log("Parallax reconstruction started")
        self.log_panel.process_started("Parallax", "Parallax")
        self.log_panel.process_snapshot(
            ProcessSnapshot(step="Parallax", parameters={"method": "Parallax", "energy": params.energy}),
        )

        operation = lambda: self.service.run_parallax(source, params)

        self.worker_thread = QThread()
        self.worker = ParallaxWorker(operation)
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
        self.log_panel.process_finished("Parallax done", elapsed_str)

        if result.rotation_degrees is not None:
            self.rotation_label.setText(f"Estimated rotation: {result.rotation_degrees:.1f}\u00b0")

        available_keys = [k for k in self.DIAGNOSTIC_KEYS if k in result.images and result.images[k] is not None]
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        self.image_selector.addItems(available_keys)
        self.image_selector.blockSignals(False)

        default_key = "Aligned BF" if "Aligned BF" in available_keys else (available_keys[0] if available_keys else None)
        if default_key is not None:
            self.image_selector.setCurrentText(default_key)
            image = result.images[default_key]
            if image is not None:
                self.workspace.update_result("selected-preview", FigureResult(f"Selected: {default_key}", np.asarray(image)))

        for name, image in result.images.items():
            if image is not None:
                self.workspace.append_result(FigureResult(f"Parallax: {name}", np.asarray(image)))

        if self.result_registry is not None:
            for name, image in result.images.items():
                key = f"parallax_{name.lower().replace(' ', '_')}"
                self.result_registry.register(
                    name=key, category="Phase Retrieval", data=image,
                    export_formats=("npy", "png", "tiff"),
                )

        self.parallax_result_ready.emit(result)
        self.workflow_state.mark_completed(self.current_process_step)

    def _handle_failed(self, error: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"Parallax reconstruction failed: {error}")
        self.log_panel.process_finished("Parallax failed", error)

    def _handle_progress(self, message: str) -> None:
        self.log_panel.log(message)
        m = re.search(r"(\d+)%", message)
        if m:
            pct = int(m.group(1))
            self.log_panel.process_progress(f"Parallax {pct}%")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.PARALLAX]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _watch_parameters(self) -> None:
        for spin in [self.energy, self.padding, self.edge_blend, self.threshold]:
            self.workflow_state.watch(spin, WorkflowStep.PARALLAX, "valueChanged")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(self, minimum, maximum, value, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "parallax_energy": self.energy.value(),
            "parallax_padding": int(self.padding.value()),
            "parallax_edge_blend": int(self.edge_blend.value()),
            "parallax_normalize": self.normalize.isChecked(),
            "parallax_threshold": self.threshold.value(),
        }
