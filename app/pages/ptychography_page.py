from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from app.services.phase_contrast_service import PhaseContrastResult
from app.services.ptychography_service import (
    BUILTIN_PROFILES, COMPUTE_PRESETS, PtychographyGeometryParams,
    PtychographyOptimizationParams, PtychographyPreprocessParams,
    PtychographyProfile, PtychographyReconstructionParams, PtychographyService,
    PtychographySetupParams, PtychographyStageResult,
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

    @Slot()
    def run(self) -> None:
        try:
            stream = ProgressStream(self.progress.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                result = self.operation()
            stream.flush()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class PtychographyPage(QWidget):
    ptychography_result_ready = Signal(object)
    operation_finished = Signal(object)
    operation_failed = Signal(str)
    operation_progress = Signal(str)

    STAGE_STEPS = {
        "data": WorkflowStep.PTYCHOGRAPHY_DATA,
        "geometry": WorkflowStep.PTYCHOGRAPHY_GEOMETRY,
        "preprocess": WorkflowStep.PTYCHOGRAPHY_PREPROCESS,
        "quick": WorkflowStep.PTYCHOGRAPHY_QUICK,
        "review": WorkflowStep.PTYCHOGRAPHY_QC,
        "optimization": WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION,
        "advanced": WorkflowStep.PTYCHOGRAPHY_ADVANCED,
        "export": WorkflowStep.PTYCHOGRAPHY_EXPORT,
    }

    def __init__(
        self, source_provider: Callable[[], object | None], log_panel: LogPanel,
        workflow_state: WorkflowState, result_registry: ResultRegistry | None = None,
        service: PtychographyService | None = None, stage_mode: str = "data",
        vacuum_probe_provider: Callable[[], object | None] | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = service or PtychographyService()
        self.vacuum_probe_provider = vacuum_probe_provider or (lambda: None)
        self.stage_mode = stage_mode
        self.worker_thread: QThread | None = None
        self.worker: PtychographyWorker | None = None
        self.pending_operation = ""
        self.pending_step = self.STAGE_STEPS[stage_mode]
        self._is_busy = False
        self.vacuum_probe_path: str | None = None
        self.custom_profile: PtychographyProfile | None = None
        self._synced_profile_id: int | None = None
        self._syncing = False
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()
        self._create_controls()
        self._build_layout()
        self._configure_stage()
        self._watch_parameters()
        self.operation_finished.connect(self._handle_worker_finished, Qt.QueuedConnection)
        self.operation_failed.connect(self._handle_worker_failed, Qt.QueuedConnection)
        self.operation_progress.connect(self._handle_worker_progress, Qt.QueuedConnection)
        self.workflow_state.changed.connect(self._refresh_stale_status)

    def _create_controls(self) -> None:
        self.profile = QComboBox()
        self.profile.addItems(list(BUILTIN_PROFILES))
        self.apply_profile_button = QPushButton("Apply Profile")
        self.apply_profile_button.clicked.connect(self.apply_profile)
        self.import_profile_button = QPushButton("Import Profile JSON")
        self.import_profile_button.clicked.connect(self.import_profile)
        self.export_profile_button = QPushButton("Export Current Profile JSON")
        self.export_profile_button.clicked.connect(self.export_profile)
        self.energy = self._float(10, 1_000_000, 80_000, 0)
        self.defocus = self._float(-1_000_000, 1_000_000, 500, 1)
        self.model = QComboBox()
        self.model.addItems(["Single-slice", "Mixed-state"])
        self.compute_preset = QComboBox()
        self.compute_preset.addItems(list(COMPUTE_PRESETS))
        self.probe_source = QComboBox()
        self.probe_source.addItems(["Ideal aperture", "Dataset role", "External file"])
        self.probe_label = QLabel("Ideal aperture initialization")
        self.probe_label.setWordWrap(True)
        self.probe_button = QPushButton("Load External Vacuum Probe")
        self.probe_button.clicked.connect(self.load_probe)
        self.data_button = QPushButton("Inspect Data & Probe")
        self.data_button.clicked.connect(self.inspect_data)
        self.suitability_label = QLabel("Run inspection to review suitability warnings.")
        self.suitability_label.setWordWrap(True)

        self.geometry_mode = QComboBox()
        self.geometry_mode.addItems(["Auto", "Use existing calibration", "Manual override"])
        self.reciprocal_sampling = self._float(0, 1000, 0.025, 6)
        self.use_reciprocal_sampling = QCheckBox("Override reciprocal sampling")
        self.scan_sampling = self._float(0, 1000, 1.0, 6)
        self.use_scan_sampling = QCheckBox("Override scan sampling")
        self.force_rotation = self._float(-360, 360, 0, 2)
        self.use_force_rotation = QCheckBox("Override CoM rotation")
        self.transpose = QCheckBox("Transpose detector coordinates")
        self.use_transpose = QCheckBox("Override transpose")
        self.semiangle = self._float(0, 1000, 20, 4)
        self.use_semiangle = QCheckBox("Override semiangle")
        self.use_semiangle.setChecked(True)
        self.probe_roi = self._int(1, 100000, 128)
        self.use_probe_roi = QCheckBox("Override probe ROI")
        self.geometry_button = QPushButton("Accept Geometry")
        self.geometry_button.clicked.connect(self.accept_geometry)

        self.vectorized_com = QCheckBox("Vectorized CoM calculation")
        self.store_initial = QCheckBox("Store initial arrays")
        self.store_initial.setChecked(True)
        self.clear_fft_cache = QCheckBox("Clear FFT cache to save memory")
        self.clear_fft_cache.setChecked(True)
        self.preprocess_batch = self._int(1, 100000, 512)
        self.preprocess_button = QPushButton("Run Preprocessing")
        self.preprocess_button.clicked.connect(self.run_preprocess)
        self.accept_preprocess_button = QPushButton("Accept Preprocessing")
        self.accept_preprocess_button.clicked.connect(self.accept_preprocessing)

        self.quick_iterations = self._int(1, 10000, 16)
        self.quick_batch = self._int(1, 100000, 256)
        self.quick_button = QPushButton("Run Quick Reconstruction")
        self.quick_button.clicked.connect(self.run_quick)

        self.qc_label = QLabel("Quick Reconstruction is required before QC.")
        self.qc_label.setWordWrap(True)
        self.qc_button = QPushButton("Calculate QC Metrics")
        self.qc_button.clicked.connect(self.run_qc)
        self.accept_qc_button = QPushButton("Confirm QC Risks")
        self.accept_qc_button.clicked.connect(self.accept_qc)

        self.optimization_method = QComboBox()
        self.optimization_method.addItems(["Grid search", "Bayesian optimization"])
        self.optimization_parameter = QComboBox()
        self.optimization_parameter.addItems(
            ["Reciprocal sampling", "Defocus", "Rotation", "Batch size", "Fix probe", "Probe modes"]
        )
        self.lower_bound = self._float(-1_000_000, 1_000_000, 0.01, 6)
        self.upper_bound = self._float(-1_000_000, 1_000_000, 0.04, 6)
        self.evaluations = self._int(2, 1000, 5)
        self.optimize_iterations = self._int(1, 512, 8)
        self.optimize_button = QPushButton("Run Optional Optimization")
        self.optimize_button.clicked.connect(self.run_optimization)
        self.apply_best_button = QPushButton("Apply Best Self-Consistency Value")
        self.apply_best_button.clicked.connect(self.apply_best_value)

        self.num_iter = self._int(1, 10000, 64)
        self.batch_size = self._int(1, 100000, 512)
        self.object_type = QComboBox()
        self.object_type.addItems(["potential", "complex"])
        self.object_positivity = QCheckBox("Enforce object positivity")
        self.object_positivity.setChecked(True)
        self.fix_probe = QCheckBox("Fix probe")
        self.seed = self._int(0, 2_147_483_647, 0)
        self.probe_modes = self._int(1, 32, 2)
        self.advanced_button = QPushButton("Run Advanced Reconstruction")
        self.advanced_button.clicked.connect(self.run_advanced)
        self.save_button = QPushButton("Save Ptychography Package")
        self.save_button.clicked.connect(self.save_package)

    def _build_layout(self) -> None:
        self.groups: dict[str, QGroupBox] = {}
        self.groups["data"] = self._group("Data & Probe", [
            ("Profile", self.profile), ("", self.apply_profile_button), ("", self.import_profile_button),
            ("", self.export_profile_button), ("Energy", self.energy), ("Defocus", self.defocus),
            ("Model", self.model), ("Compute preset", self.compute_preset), ("Probe source", self.probe_source),
            ("", self.probe_button), ("", self.probe_label), ("", self.data_button), ("", self.suitability_label),
        ])
        self.groups["geometry"] = self._group("Calibration / Geometry", [
            ("Mode", self.geometry_mode), ("", self.use_reciprocal_sampling),
            ("Reciprocal sampling", self.reciprocal_sampling), ("", self.use_scan_sampling),
            ("Scan sampling", self.scan_sampling), ("", self.use_force_rotation),
            ("CoM rotation", self.force_rotation), ("", self.use_transpose), ("", self.transpose),
            ("", self.use_semiangle), ("Semiangle", self.semiangle), ("", self.use_probe_roi),
            ("Probe ROI", self.probe_roi), ("", self.geometry_button),
        ])
        self.groups["preprocess"] = self._group("Preprocess", [
            ("", self.vectorized_com), ("", self.store_initial), ("", self.clear_fft_cache),
            ("Max batch", self.preprocess_batch), ("", self.preprocess_button),
            ("", self.accept_preprocess_button),
        ])
        self.groups["quick"] = self._group("Quick Reconstruction", [
            ("Iterations", self.quick_iterations), ("Safe batch", self.quick_batch), ("", self.quick_button),
        ])
        self.groups["review"] = self._group("Review & QC", [
            ("", self.qc_label), ("", self.qc_button), ("", self.accept_qc_button),
        ])
        self.groups["optimization"] = self._group("Optional Parameter Optimization", [
            ("Method", self.optimization_method), ("Parameter", self.optimization_parameter),
            ("Lower bound", self.lower_bound), ("Upper bound", self.upper_bound),
            ("Evaluations", self.evaluations), ("Diagnostic iterations", self.optimize_iterations),
            ("", self.optimize_button), ("", self.apply_best_button),
        ])
        self.groups["advanced"] = self._group("Advanced Reconstruction", [
            ("Iterations", self.num_iter), ("Max batch", self.batch_size), ("Object type", self.object_type),
            ("Random seed", self.seed), ("Probe modes", self.probe_modes), ("", self.object_positivity),
            ("", self.fix_probe), ("", self.advanced_button),
        ])
        self.groups["export"] = self._group("Export", [("", self.save_button)])
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        for group in self.groups.values():
            layout.addWidget(group)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.controls_panel = controls
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.workspace)

    @staticmethod
    def _group(title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        for label, widget in rows:
            form.addRow(label, widget)
        return group

    def _configure_stage(self) -> None:
        for stage, group in self.groups.items():
            group.setVisible(stage == self.stage_mode)
        self.refresh_stage()

    def refresh_stage(self, *_args) -> None:
        ctx = self.service.context
        self._sync_from_context()
        enabled = not self._is_busy
        self.data_button.setEnabled(enabled)
        self.geometry_button.setEnabled(enabled and ctx.data_result is not None)
        self.preprocess_button.setEnabled(enabled and ctx.geometry_result is not None)
        self.accept_preprocess_button.setEnabled(enabled and ctx.preprocess_result is not None)
        self.quick_button.setEnabled(enabled and ctx.preprocessing_accepted)
        self.qc_button.setEnabled(enabled and ctx.quick_result is not None)
        self.accept_qc_button.setEnabled(enabled and ctx.qc_result is not None)
        self.optimize_button.setEnabled(enabled and ctx.preprocessing_accepted)
        self.apply_best_button.setEnabled(enabled and ctx.optimization_result is not None)
        self.advanced_button.setEnabled(enabled and ctx.preprocessing_accepted and ctx.qc_accepted)
        self.save_button.setEnabled(enabled and ctx.advanced_result is not None)
        result = {
            "data": ctx.data_result, "geometry": ctx.geometry_result, "preprocess": ctx.preprocess_result,
            "quick": ctx.quick_result, "review": ctx.qc_result, "optimization": ctx.optimization_result,
            "advanced": ctx.advanced_result,
        }.get(self.stage_mode)
        if result is not None:
            self._show_result(result)

    def apply_profile(self) -> None:
        selected = self.custom_profile if self.profile.currentText() == "Custom" else self.profile.currentText()
        if selected is None:
            return
        profile = self.service.apply_profile(selected)
        self._apply_profile_controls(profile)
        self.workflow_state.parameters_updated(WorkflowStep.PTYCHOGRAPHY_DATA)
        self.status_label.setText(f"Applied profile: {profile.name}.")
        self.refresh_stage()

    def import_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Ptychography Profile", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.custom_profile = self.service.load_profile(Path(path))
        except Exception as exc:
            self._failed(str(exc))
            return
        if self.profile.findText("Custom") < 0:
            self.profile.addItem("Custom")
        self.profile.setCurrentText("Custom")
        self.apply_profile()

    def export_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Ptychography Profile", "ptychography_profile.json",
                                              "JSON (*.json)")
        if path:
            self.service.save_profile(self._current_profile(), Path(path))
            self.status_label.setText("Profile JSON saved.")

    def load_probe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Vacuum Probe", "", "HDF5 (*.h5 *.hdf5);;All Files (*)")
        if path:
            self.vacuum_probe_path = path
            self.probe_source.setCurrentText("External file")
            self.probe_label.setText(f"Probe: {Path(path).name}")
            self._parameters_changed(WorkflowStep.PTYCHOGRAPHY_DATA)

    def inspect_data(self) -> None:
        source = self._source()
        if source is not None:
            self._finish_sync("Data & Probe", lambda: self.service.inspect_data_probe(
                source, self._setup_params(), self._probe_override()
            ),
                              WorkflowStep.PTYCHOGRAPHY_DATA)

    def accept_geometry(self) -> None:
        self._finish_sync("Calibration / Geometry", lambda: self.service.set_geometry(self._geometry_params()),
                          WorkflowStep.PTYCHOGRAPHY_GEOMETRY)

    def run_preprocess(self) -> None:
        source = self._source()
        if source is not None:
            self._start("Ptychography Preprocessing",
                        lambda: self.service.preprocess(
                            source,
                            self.service.context.setup_params or self._setup_params(),
                            self._preprocess_params(),
                            self.service.context.geometry_params or self._geometry_params(),
                            self._probe_override(),
                        ),
                        WorkflowStep.PTYCHOGRAPHY_PREPROCESS)

    def accept_preprocessing(self) -> None:
        try:
            self.service.accept_preprocessing()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.workflow_state.mark_completed(WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT)
        self.status_label.setText("Preprocessing accepted. Quick Reconstruction is ready.")
        self.refresh_stage()

    def run_quick(self) -> None:
        self._start("Quick Reconstruction", lambda: self.service.quick_reconstruct(self._quick_params()),
                    WorkflowStep.PTYCHOGRAPHY_QUICK)

    def run_qc(self) -> None:
        self._finish_sync("Review & QC", self.service.review_qc, WorkflowStep.PTYCHOGRAPHY_QC)

    def accept_qc(self) -> None:
        try:
            self.service.accept_qc()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.workflow_state.mark_completed(WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT)
        self.status_label.setText("QC risks confirmed. Advanced Reconstruction is ready.")
        self.refresh_stage()

    def run_optimization(self) -> None:
        self._start("Ptychography Optimization", lambda: self.service.optimize(self._optimization_params()),
                    WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION)

    def apply_best_value(self) -> None:
        result = self.service.context.optimization_result
        if result is None:
            return
        value = float(result.metadata["best_value"])
        parameter = self.optimization_parameter.currentText()
        if parameter == "Reciprocal sampling":
            self.reciprocal_sampling.setValue(value)
            self.use_reciprocal_sampling.setChecked(True)
        elif parameter == "Defocus":
            self.defocus.setValue(value)
        elif parameter == "Rotation":
            self.force_rotation.setValue(value)
            self.use_force_rotation.setChecked(True)
        elif parameter == "Batch size":
            self.batch_size.setValue(value)
        elif parameter == "Fix probe":
            self.fix_probe.setChecked(value >= 0.5)
        elif parameter == "Probe modes":
            self.probe_modes.setValue(value)
        self.service.invalidate_from("preprocess")
        self.workflow_state.parameters_updated(WorkflowStep.PTYCHOGRAPHY_GEOMETRY)
        self.status_label.setText(f"Applied {parameter}: {value:.6g}. Re-run and re-accept downstream stages.")
        self.refresh_stage()

    def run_advanced(self) -> None:
        self._start("Advanced Reconstruction", lambda: self.service.advanced_reconstruct(self._advanced_params()),
                    WorkflowStep.PTYCHOGRAPHY_ADVANCED)

    def save_package(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Save Ptychography Package")
        if not directory:
            return
        try:
            saved = self.service.save_package(Path(directory))
        except Exception as exc:
            self._failed(str(exc))
            return
        self.workflow_state.mark_completed(WorkflowStep.PTYCHOGRAPHY_EXPORT)
        self.status_label.setText(f"Saved {len(saved)} package files.")

    def _finish_sync(self, name: str, operation, step: str) -> None:
        try:
            result = operation()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.pending_operation, self.pending_step = name, step
        self._complete_result(result)

    def _start(self, name: str, operation, step: str) -> None:
        if self._is_busy or self.worker_thread is not None:
            self.status_label.setText("A Ptychography operation is already running.")
            return
        self.pending_operation, self.pending_step = name, step
        self._is_busy = True
        self.status_label.setText(f"Running {name}...")
        self.refresh_stage()
        self.log_panel.process_started(name, name)
        self.log_panel.process_snapshot(ProcessSnapshot(step=name, parameters=self.params_snapshot()))
        thread = QThread(self)
        worker = PtychographyWorker(operation)
        self.worker_thread, self.worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.operation_finished.emit)
        worker.failed.connect(self.operation_failed.emit)
        worker.progress.connect(self.operation_progress.emit)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_refs, Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(object)
    def _handle_worker_finished(self, result) -> None:
        self._complete_result(result)

    @Slot(str)
    def _handle_worker_failed(self, error: str) -> None:
        self._failed(error)

    @Slot(str)
    def _handle_worker_progress(self, message: str) -> None:
        self.status_label.setText(message)
        self.log_panel.process_progress(message)

    def _complete_result(self, result: PtychographyStageResult | PhaseContrastResult) -> None:
        self._show_result(result)
        self._register(result.images, getattr(result, "metadata", {}))
        if result.stage == "data" if isinstance(result, PtychographyStageResult) else False:
            warnings = result.metadata.get("warnings", [])
            self.suitability_label.setText("\n".join(warnings) if warnings else "No automatic suitability warnings.")
        if isinstance(result, PtychographyStageResult) and result.stage == "qc":
            metrics = result.metadata.get("metrics", {})
            warnings = result.metadata.get("warnings", [])
            text = "\n".join(f"{key}: {value:.4g}" for key, value in metrics.items())
            self.qc_label.setText(text + ("\nWarnings:\n" + "\n".join(warnings) if warnings else "\nNo threshold warnings."))
        self.status_label.setText(f"{self.pending_operation} complete.")
        self.log_panel.process_finished(self.pending_operation, f"elapsed={result.elapsed_seconds:.2f}s")
        self.workflow_state.mark_completed(self.pending_step)
        if isinstance(result, PhaseContrastResult) and self.pending_step == WorkflowStep.PTYCHOGRAPHY_ADVANCED:
            self.workflow_state.mark_completed(WorkflowStep.PTYCHOGRAPHY)
            self.ptychography_result_ready.emit(result)
        self.refresh_stage()

    def _show_result(self, result: PtychographyStageResult | PhaseContrastResult) -> None:
        figures = [
            FigureResult(name, np.asarray(image), scaling="linear" if "error" in name.lower() else "log")
            for name, image in result.images.items()
            if np.asarray(image).ndim >= 2
        ]
        self.workspace.set_results(figures[:8])

    def _register(self, images: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
        if self.result_registry is None:
            return
        safe = {key: str(value) for key, value in metadata.items()}
        prefix = f"ptychography_{self.pending_step}_"
        for name, image in images.items():
            self.result_registry.register(prefix + name.lower().replace(" ", "_"), "Phase Retrieval",
                                          image, ("npy", "png", "tiff"), safe)

    def _source(self):
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Ptychography", "Load and assign a Target DataCube first.")
        return source

    @Slot(str)
    def _failed(self, error: str) -> None:
        self.status_label.setText(f"Failed: {error}")
        self.log_panel.log(f"Ptychography failed: {error}")
        self.log_panel.process_finished(self.pending_operation or "Ptychography failed", error)
        self.refresh_stage()

    @Slot()
    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._is_busy = False
        self.refresh_stage()

    def _setup_params(self) -> PtychographySetupParams:
        path = self.vacuum_probe_path if self.probe_source.currentText() == "External file" else None
        return PtychographySetupParams(self.energy.value(), self.defocus.value(), self.model.currentText(),
                                       self.compute_preset.currentText(), path,
                                       self.probe_source.currentText())

    def _probe_override(self) -> object | None:
        return self.vacuum_probe_provider() if self.probe_source.currentText() == "Dataset role" else None

    def _geometry_params(self) -> PtychographyGeometryParams:
        return PtychographyGeometryParams(
            self.geometry_mode.currentText(),
            self.reciprocal_sampling.value() if self.use_reciprocal_sampling.isChecked() else None,
            self.scan_sampling.value() if self.use_scan_sampling.isChecked() else None,
            self.force_rotation.value() if self.use_force_rotation.isChecked() else None,
            self.transpose.isChecked() if self.use_transpose.isChecked() else None,
            self.semiangle.value() if self.use_semiangle.isChecked() else None,
            int(self.probe_roi.value()) if self.use_probe_roi.isChecked() else None,
        )

    def _preprocess_params(self) -> PtychographyPreprocessParams:
        return PtychographyPreprocessParams(self.vectorized_com.isChecked(), self.store_initial.isChecked(),
                                            int(self.preprocess_batch.value()), self.clear_fft_cache.isChecked())

    def _quick_params(self) -> PtychographyReconstructionParams:
        profile = self.service.context.active_profile.quick
        return replace(
            profile, num_iter=int(self.quick_iterations.value()), max_batch_size=int(self.quick_batch.value()),
            object_type="complex", object_positivity=False, fix_probe=False, seed_random=0, num_probe_modes=1,
        )

    def _advanced_params(self) -> PtychographyReconstructionParams:
        return PtychographyReconstructionParams(
            int(self.num_iter.value()), int(self.batch_size.value()), self.object_type.currentText(),
            self.object_positivity.isChecked(), self.fix_probe.isChecked(), int(self.seed.value()),
            int(self.probe_modes.value()),
        )

    def _optimization_params(self) -> PtychographyOptimizationParams:
        return PtychographyOptimizationParams(
            self.optimization_method.currentText(), self.optimization_parameter.currentText(),
            self.lower_bound.value(), self.upper_bound.value(), int(self.evaluations.value()),
            int(self.optimize_iterations.value()),
        )

    def _current_profile(self) -> PtychographyProfile:
        return PtychographyProfile("Custom", self._setup_params(), self._geometry_params(),
                                   self._preprocess_params(), self._quick_params(),
                                   self._optimization_params(), self._advanced_params(),
                                   self.service.context.active_profile.qc)

    def _apply_profile_controls(self, profile: PtychographyProfile) -> None:
        setup, geometry, preprocess, quick, advanced = (
            profile.setup, profile.geometry, profile.preprocess, profile.quick, profile.advanced
        )
        self.energy.setValue(setup.energy)
        self.defocus.setValue(setup.defocus)
        self.model.setCurrentText(setup.model)
        self.compute_preset.setCurrentText(setup.compute_preset)
        self.geometry_mode.setCurrentText(geometry.mode)
        if geometry.semiangle_cutoff is not None:
            self.semiangle.setValue(geometry.semiangle_cutoff)
            self.use_semiangle.setChecked(True)
        self.preprocess_batch.setValue(preprocess.max_batch_size)
        self.vectorized_com.setChecked(preprocess.vectorized_com_calculation)
        self.clear_fft_cache.setChecked(preprocess.clear_fft_cache)
        self.quick_iterations.setValue(quick.num_iter)
        self.quick_batch.setValue(quick.max_batch_size)
        self.num_iter.setValue(advanced.num_iter)
        self.batch_size.setValue(advanced.max_batch_size)
        self.object_type.setCurrentText(advanced.object_type)
        self.object_positivity.setChecked(advanced.object_positivity)
        self.fix_probe.setChecked(advanced.fix_probe)
        self.probe_modes.setValue(advanced.num_probe_modes)

    def _sync_from_context(self) -> None:
        profile = self.service.context.active_profile
        if self._synced_profile_id != id(profile):
            self._syncing = True
            self._apply_profile_controls(profile)
            self._syncing = False
            self._synced_profile_id = id(profile)
        setup = self.service.context.setup_params
        if setup is not None:
            for control, value in ((self.energy, setup.energy), (self.defocus, setup.defocus)):
                control.blockSignals(True); control.setValue(value); control.blockSignals(False)
            for control, value in ((self.model, setup.model), (self.compute_preset, setup.compute_preset),
                                   (self.probe_source, setup.probe_source)):
                control.blockSignals(True); control.setCurrentText(value); control.blockSignals(False)
            self.vacuum_probe_path = setup.vacuum_probe_path

    def set_cuda_enabled(self, _enabled: bool) -> None:
        # CUDA availability is exposed as a choice; profiles never switch compute mode silently.
        return

    def _watch_parameters(self) -> None:
        step = self.STAGE_STEPS[self.stage_mode]
        for control in self.findChildren(NumericLineEdit):
            control.valueChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QComboBox):
            control.currentTextChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QCheckBox):
            control.toggled.connect(lambda *_args, s=step: self._parameters_changed(s))

    def _parameters_changed(self, step: str) -> None:
        if self._syncing:
            return
        if step == WorkflowStep.PTYCHOGRAPHY_DATA:
            self.service.invalidate_from("geometry")
        elif step == WorkflowStep.PTYCHOGRAPHY_GEOMETRY:
            self.service.invalidate_from("preprocess")
        elif step == WorkflowStep.PTYCHOGRAPHY_PREPROCESS:
            self.service.invalidate_from("quick")
        elif step == WorkflowStep.PTYCHOGRAPHY_QUICK:
            self.service.invalidate_from("quick")
        elif step == WorkflowStep.PTYCHOGRAPHY_QC:
            self.service.context.qc_accepted = False
        elif step == WorkflowStep.PTYCHOGRAPHY_ADVANCED:
            self.service.invalidate_from("advanced")
        self.workflow_state.parameters_updated(step)
        self.refresh_stage()

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.is_stale(self.STAGE_STEPS[self.stage_mode]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def params_snapshot(self) -> dict[str, object]:
        profile = self._current_profile()
        return {"stage": self.stage_mode, "ptychography_profile": profile.name,
                "ptychography_profile_data": {
                    "setup": profile.setup.__dict__, "geometry": profile.geometry.__dict__,
                    "preprocess": profile.preprocess.__dict__, "quick": profile.quick.__dict__,
                    "optimization": profile.optimization.__dict__, "advanced": profile.advanced.__dict__,
                    "qc": profile.qc.__dict__,
                }}

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        data = params.get("ptychography_profile_data")
        if not isinstance(data, dict):
            return
        profile = PtychographyProfile(
            str(params.get("ptychography_profile", "Restored")),
            PtychographySetupParams(**data.get("setup", {})),
            PtychographyGeometryParams(**data.get("geometry", {})),
            PtychographyPreprocessParams(**data.get("preprocess", {})),
            PtychographyReconstructionParams(**data.get("quick", {})),
            PtychographyOptimizationParams(**data.get("optimization", {})),
            PtychographyReconstructionParams(**data.get("advanced", {})),
            self.service.context.active_profile.qc,
        )
        self.custom_profile = profile
        self.service.context.active_profile = profile
        self._apply_profile_controls(profile)

    @staticmethod
    def _float(minimum, maximum, value, decimals) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals)

    @staticmethod
    def _int(minimum, maximum, value) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, integer=True)
