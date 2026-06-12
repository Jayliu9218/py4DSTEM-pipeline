from __future__ import annotations

import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
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
    PtychographyParams,
    PhaseContrastResult,
    PhaseContrastService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.progress_stream import ProgressStream


class PtychographyWorker(QObject):
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


class PtychographyPage(QWidget):
    ptychography_result_ready = Signal(object)

    DIAGNOSTIC_KEYS = [
        "Phase",
        "Amplitude",
        "Probe Intensity",
        "Fourier Probe",
    ]

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
        dpc_result_provider: Callable[[], PhaseContrastResult | None] | None = None,
        parallax_result_provider: Callable[[], PhaseContrastResult | None] | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.dpc_result_provider = dpc_result_provider
        self.parallax_result_provider = parallax_result_provider
        self.service = PhaseContrastService()
        self.worker_thread: QThread | None = None
        self.worker: PtychographyWorker | None = None
        self.current_process_step = WorkflowStep.PTYCHOGRAPHY
        self.result: PhaseContrastResult | None = None
        self.vacuum_probe_path: str | None = None

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        self.energy = self._float_input(10, 1000000, 80000, decimals=0, unit="eV")
        self.defocus = self._float_input(-1000000, 1000000, 500, decimals=1, unit="A")
        self.vacuum_probe_label = QLabel("No vacuum probe loaded")
        self.vacuum_probe_btn = QPushButton("Load Vacuum Probe")
        self.vacuum_probe_btn.clicked.connect(self._load_vacuum_probe)
        self.num_iter = self._int_input(1, 512, 64, unit="iterations")
        self.batch_size = self._int_input(1, 4096, 512, unit="positions")
        self.object_type = QComboBox()
        self.object_type.addItems(["potential", "complex"])
        self.positivity_label = QLabel("object_positivity: enabled")

        self.image_selector = QComboBox()
        self.image_selector.addItems(self.DIAGNOSTIC_KEYS)
        self.image_selector.currentTextChanged.connect(self._show_selected_image)

        self.rotation_label = QLabel("")
        self.rotation_label.setWordWrap(True)

        self.compare_button = QPushButton("Compare with DPC / Parallax")
        self.compare_button.clicked.connect(self._show_comparison)

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
        input_form.addRow("Defocus", self.defocus)
        input_form.addRow(self.vacuum_probe_btn)
        input_form.addRow(self.vacuum_probe_label)

        recon_group = QGroupBox("2. Reconstruction")
        recon_form = QFormLayout(recon_group)
        recon_form.addRow("Iterations", self.num_iter)
        recon_form.addRow("Max Batch", self.batch_size)
        recon_form.addRow("Object Type", self.object_type)
        recon_form.addRow(self.positivity_label)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(input_group)
        layout.addWidget(recon_group)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("View result:"))
        layout.addWidget(self.image_selector)
        layout.addWidget(self.rotation_label)
        layout.addWidget(self.compare_button)
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

    def _show_comparison(self) -> None:
        has_any = False
        dpc = self.dpc_result_provider() if self.dpc_result_provider else None
        parallax = self.parallax_result_provider() if self.parallax_result_provider else None
        ptychography = self.result

        if ptychography is not None:
            phase = ptychography.images.get("Phase")
            if phase is not None:
                self.workspace.update_result("comparison-ptychography", FigureResult("Ptychography Phase", np.asarray(phase)))
                has_any = True

        if parallax is not None:
            aligned = parallax.images.get("Aligned BF")
            if aligned is not None:
                self.workspace.update_result("comparison-parallax", FigureResult("Parallax Aligned BF", np.asarray(aligned)))
                has_any = True

        if dpc is not None:
            com = dpc.images.get("Complex CoM")
            if com is None:
                com = dpc.images.get("Phase")
            if com is not None:
                self.workspace.update_result("comparison-dpc", FigureResult("DPC / CoM", np.asarray(com)))
                has_any = True

        if not has_any:
            QMessageBox.information(self, "Ptychography Comparison",
                "No phase retrieval results available. Run DPC, Parallax, or Ptychography first.")

    def _load_vacuum_probe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Vacuum Probe", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if path:
            self.vacuum_probe_path = path
            self.vacuum_probe_label.setText(f"Probe: {Path(path).name}")

    def _current_params(self) -> PtychographyParams:
        return PtychographyParams(
            energy=self.energy.value(),
            defocus=self.defocus.value(),
            vacuum_probe_path=self.vacuum_probe_path,
            num_iter=int(self.num_iter.value()),
            max_batch_size=int(self.batch_size.value()),
            object_type=self.object_type.currentText(),
        )

    def _run(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Ptychography", "Load a 4D DataCube first.")
            return

        if not self.workflow_state.is_completed(WorkflowStep.PARALLAX):
            result = QMessageBox.question(
                self, "Ptychography",
                "Recommended: run Parallax first to estimate defocus and detector rotation.\nContinue with manual/default parameters?",
            )
            if result != QMessageBox.Yes:
                return

        params = self._current_params()
        self.status_label.setText("Running Ptychography...")
        self.run_button.setEnabled(False)
        self.log_panel.log("Ptychography reconstruction started")
        self.log_panel.process_started("Ptychography", "Ptychography")
        self.log_panel.process_snapshot(
            ProcessSnapshot(step="Ptychography", parameters={"method": "Ptychography", "energy": params.energy}),
        )

        operation = lambda: self.service.run_ptychography(source, params)

        self.worker_thread = QThread()
        self.worker = PtychographyWorker(operation)
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
        self.log_panel.process_finished("Ptychography done", elapsed_str)

        if result.rotation_degrees is not None:
            self.rotation_label.setText(f"Estimated rotation: {result.rotation_degrees:.1f}\u00b0")

        available_keys = [k for k in self.DIAGNOSTIC_KEYS if k in result.images and result.images[k] is not None]
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        self.image_selector.addItems(available_keys)
        self.image_selector.blockSignals(False)

        default_key = "Phase" if "Phase" in available_keys else (available_keys[0] if available_keys else None)
        if default_key is not None:
            self.image_selector.setCurrentText(default_key)
            image = result.images[default_key]
            if image is not None:
                self.workspace.update_result("selected-preview", FigureResult(f"Selected: {default_key}", np.asarray(image)))

        self.workspace.append_results([
            FigureResult(f"Ptychography: {name}", np.asarray(image))
            for name, image in result.images.items()
            if image is not None
        ])

        if self.result_registry is not None:
            for name, image in result.images.items():
                key = f"ptychography_{name.lower().replace(' ', '_')}"
                self.result_registry.register(
                    name=key, category="Phase Retrieval", data=image,
                    export_formats=("npy", "png", "tiff"),
                )

        self.ptychography_result_ready.emit(result)
        self.workflow_state.mark_completed(self.current_process_step)

    def _handle_failed(self, error: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"Ptychography reconstruction failed: {error}")
        self.log_panel.process_finished("Ptychography failed", error)

    def _handle_progress(self, message: str) -> None:
        self.log_panel.log(message)
        m = re.search(r"(\d+)%", message)
        if m:
            pct = int(m.group(1))
            self.log_panel.process_progress(f"Ptychography {pct}%")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.PTYCHOGRAPHY]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _watch_parameters(self) -> None:
        for spin in [self.energy, self.defocus, self.num_iter, self.batch_size]:
            self.workflow_state.watch(spin, WorkflowStep.PTYCHOGRAPHY, "valueChanged")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(self, minimum, maximum, value, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "ptychography_energy": self.energy.value(),
            "ptychography_defocus": self.defocus.value(),
            "ptychography_num_iter": int(self.num_iter.value()),
            "ptychography_batch_size": int(self.batch_size.value()),
            "ptychography_object_type": self.object_type.currentText(),
            "ptychography_vacuum_probe_path": self.vacuum_probe_path,
        }
