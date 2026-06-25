from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.services.phase_mapping_service import (
    PhaseMatchResult, PhaseMatchParams, PhasePlanParams, PhaseMappingService,
    PhaseMappingServiceError,
)
from app.services.crystal_analysis_service import (
    CrystalAnalysisRunConfig,
    CrystalAnalysisService,
    CrystalAnalysisStageResult,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class StructuralPhasePage(QWidget, WorkerRunner):
    phase_result_ready = Signal(object)

    STAGE_STEPS = {
        "library": WorkflowStep.STRUCTURAL_PHASE_PLAN,
        "match": WorkflowStep.STRUCTURAL_PHASE_MATCH,
        "results": WorkflowStep.STRUCTURAL_PHASE,
        "structure": WorkflowStep.CRYSTAL_STRUCTURE_FACTORS,
        "simulated": WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
        "orientation": WorkflowStep.CRYSTAL_ORIENTATION,
        "grain": WorkflowStep.CRYSTAL_GRAIN,
        "strain": WorkflowStep.CRYSTAL_STRAIN,
    }

    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
        service: PhaseMappingService | None = None,
        stage_mode: str = "library",
        workspace: AdaptiveImageWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = service or PhaseMappingService()
        self.stage_mode = stage_mode
        self.cuda_enabled = False
        self._init_worker_runner()
        self.pending_name = ""
        self.workspace = workspace or AdaptiveImageWorkspace()
        self.status_label = QLabel("Phase mapping workflow ready.")
        self._create_controls()
        self._build_layout()
        self._configure_stage()
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)

    def _create_controls(self) -> None:
        self.crystal_list = QListWidget()
        self.crystal_list.setMaximumHeight(140)
        self.add_cif_button = QPushButton("Add Crystal CIF")
        self.add_cif_button.clicked.connect(self.add_crystal)
        self.remove_crystal_button = QPushButton("Remove Selected")
        self.remove_crystal_button.clicked.connect(self.remove_crystal)
        self.run_mode = QComboBox()
        self.run_mode.addItems(["ROI 128x128", "Full Dataset"])
        self.roi_size = self._int(16, 100000, 128)

        self.mode = QComboBox()
        self.mode.addItems(["General 3D", "Fiber"])
        self.voltage = self._float(1000, 1_000_000, 300_000, 0)
        self.k_max = self._float(0.1, 10, 1.5, 3)
        self.zone_step = self._float(0.1, 30, 2, 2)
        self.plane_step = self._float(0.1, 30, 2, 2)
        self.corr_kernel_size = self._float(0.001, 10, 0.08, 4)
        self.sigma = self._float(0.001, 10, 0.02, 4)
        self.fiber_x = self._float(-100, 100, 0, 3)
        self.fiber_y = self._float(-100, 100, 0, 3)
        self.fiber_z = self._float(-100, 100, 1, 3)
        self.fiber_start = self._float(-360, 360, 0, 2)
        self.fiber_end = self._float(-360, 360, 360, 2)
        self.symmetry_order = self._int(1, 24, 6)
        self.plan_button = QPushButton("Create Multi-Phase Plan")
        self.plan_button.clicked.connect(
            lambda: self._start("Multi-Phase Plan", lambda: self.service.create_multi_phase_plan(self._plan_params()))
        )
        self.structure_button = QPushButton("Calculate Structure Factors")
        self.structure_button.clicked.connect(
            lambda: self._start(
                "Structure Factors",
                lambda: self._crystal_service().calculate_structure_factors(self._plan_params()),
            )
        )
        self.simulate_button = QPushButton("Build Simulated Diffraction Library")
        self.simulate_button.clicked.connect(
            lambda: self._start(
                "Simulated Diffraction",
                lambda: self._crystal_service().build_orientation_libraries(self._plan_params()),
            )
        )

        self.match_matches = self._int(1, 20, 2)
        self.match_min_angle = self._float(0, 180, 5, 2)
        self.match_min_peaks = self._int(1, 1000, 3)
        self.match_inversion = QCheckBox("Use inversion symmetry")
        self.match_inversion.setChecked(True)
        self.corr_normalize = QCheckBox("Normalize correlation")
        self.corr_normalize.setChecked(True)
        self.low_confidence = self._float(-1000, 1000, 0.1, 4)
        self.match_button = QPushButton("Match All Phases")
        self.match_button.clicked.connect(
            lambda: self._start("Phase Matching", lambda: self.service.match_phases(self.braggvectors_provider(), self._match_params()))
        )
        self.orientation_button = QPushButton("Review Phase-Conditioned Orientation")
        self.orientation_button.clicked.connect(
            lambda: self._start(
                "Orientation Matching",
                lambda: self._crystal_service().orientation_summary(),
            )
        )
        self.grain_button = QPushButton("Run Grain Analysis")
        self.grain_button.clicked.connect(
            lambda: self._start(
                "Grain Analysis",
                lambda: self._crystal_service().run_grain_analysis(),
            )
        )
        self.strain_button = QPushButton("Run Crystal Strain Analysis")
        self.strain_button.clicked.connect(
            lambda: self._start(
                "Crystal Strain Analysis",
                lambda: self._crystal_service().run_strain_analysis(self.braggvectors_provider()),
            )
        )

    def _build_layout(self) -> None:
        self.groups: dict[str, QGroupBox] = {}
        self.groups["library"] = self._group("Crystal Library", [
            ("Analysis mode", self.run_mode),
            ("ROI size", self.roi_size),
            ("Loaded crystals", self.crystal_list),
            ("", self._row(self.add_cif_button, self.remove_crystal_button)),
        ])
        self.groups["plan"] = self._group("Multi-Phase Orientation Plan", [
            ("Mode", self.mode), ("Accelerating voltage", self.voltage), ("k_max", self.k_max),
            ("Zone-axis step", self.zone_step), ("In-plane step", self.plane_step),
            ("Correlation kernel", self.corr_kernel_size), ("Excitation sigma", self.sigma),
            ("Fiber axis x", self.fiber_x), ("Fiber axis y", self.fiber_y), ("Fiber axis z", self.fiber_z),
            ("Fiber angle start", self.fiber_start), ("Fiber angle end", self.fiber_end),
            ("Fiber symmetry order", self.symmetry_order), ("", self.plan_button),
        ])
        self.groups["structure"] = self._group("Structure Factors", [
            ("Parameters", QLabel("Uses voltage and k_max from the CIF planning controls.")),
            ("", self.structure_button),
        ])
        self.groups["simulated"] = self._group("Simulated Diffraction", [
            ("Parameters", QLabel("Uses mode, angular steps, and kernel settings from the CIF plan.")),
            ("", self.simulate_button),
        ])
        self.groups["match"] = self._group("Phase Matching", [
            ("Candidates", self.match_matches), ("Min candidate angle", self.match_min_angle),
            ("Min peaks", self.match_min_peaks), ("Low-confidence threshold", self.low_confidence),
            ("", self.match_inversion), ("", self.corr_normalize), ("", self.match_button),
        ])
        self.groups["orientation"] = self._group("Orientation Matching", [
            ("", self.orientation_button),
        ])
        self.groups["grain"] = self._group("Grain Analysis", [
            ("", self.grain_button),
        ])
        self.groups["strain"] = self._group("Strain Analysis", [
            ("", self.strain_button),
        ])
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        for group in self.groups.values():
            layout.addWidget(group)
        layout.addStretch(1)
        self.controls_panel = controls
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.workspace)

    def _configure_stage(self) -> None:
        visible = {
            "library": {"library", "plan"},
            "structure": {"library", "structure"},
            "simulated": {"library", "simulated"},
            "match": {"match"},
            "orientation": {"orientation"},
            "grain": {"grain"},
            "strain": {"strain"},
            "results": {},
        }[self.stage_mode]
        for name, group in self.groups.items():
            group.setVisible(name in visible)
        self.refresh_stage()

    def refresh_stage(self, *_args) -> None:
        self._update_crystal_list()
        self.plan_button.setEnabled(bool(self.service.enabled_crystals()))
        self.structure_button.setEnabled(bool(self.service.enabled_crystals()))
        self.simulate_button.setEnabled(bool(self.service.enabled_crystals()))
        self.match_button.setEnabled(self.service.plan_ready)
        phase_ready = self.service.result is not None
        self.orientation_button.setEnabled(phase_ready)
        self.grain_button.setEnabled(phase_ready)
        self.strain_button.setEnabled(phase_ready)

    def _update_crystal_list(self) -> None:
        self.crystal_list.clear()
        for entry in self.service.crystals:
            label = entry.source + ("" if entry.enabled else "  (disabled)")
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if entry.enabled else Qt.Unchecked)
            self.crystal_list.addItem(item)

    def add_crystal(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add crystal structure", "", "CIF files (*.cif)")
        if not path:
            return
        try:
            self.service.load_crystal(path)
            self._update_crystal_list()
            self.workflow_state.parameters_updated(WorkflowStep.STRUCTURAL_PHASE_PLAN)
            self._notify(f"Crystal loaded: {Path(path).name}. Create the multi-phase plan.", "success")
        except PhaseMappingServiceError as exc:
            self._failed(str(exc))

    def remove_crystal(self) -> None:
        row = self.crystal_list.currentRow()
        if row < 0:
            return
        try:
            self.service.remove_crystal(row)
            self._update_crystal_list()
            self.workflow_state.parameters_updated(WorkflowStep.STRUCTURAL_PHASE_PLAN)
            self._notify("Crystal removed. Recreate the multi-phase plan.", "info")
        except PhaseMappingServiceError as exc:
            self._failed(str(exc))

    def _start(self, name: str, operation) -> None:
        if name == "Phase Matching":
            self.workspace.lock_auto_layout()
        self._sync_run_config()
        self.pending_name = name
        self._start_background(name, lambda _cb: operation(), parameters=self.params_snapshot())

    def _handle_result(self, result) -> None:
        if isinstance(result, float):
            self._notify(f"{self.pending_name} complete in {result:.2f}s.", "success")
            self.log_panel.process_finished(self.pending_name, self.status_label.text())
            self.workflow_state.mark_completed(WorkflowStep.STRUCTURAL_PHASE_PLAN)
            self.refresh_stage()
            return
        if isinstance(result, CrystalAnalysisStageResult):
            figures = [
                FigureResult(name, image, image_kind="rgb" if np.asarray(image).ndim == 3 else "intensity")
                for name, image in result.images.items()
            ]
            if figures:
                self.workspace.set_results(figures)
            detail = ", ".join(f"{key}={value}" for key, value in result.metrics.items())
            warning = " ".join(result.warnings)
            self._notify(
                f"{self.pending_name} complete in {result.elapsed_seconds:.2f}s. {detail} {warning}".strip(),
                "warning" if result.warnings else "success",
            )
            self.log_panel.process_finished(self.pending_name, self.status_label.text())
            step = {
                "structure": WorkflowStep.CRYSTAL_STRUCTURE_FACTORS,
                "simulated": WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
                "orientation": WorkflowStep.CRYSTAL_ORIENTATION,
                "grain": WorkflowStep.CRYSTAL_GRAIN,
                "strain": WorkflowStep.CRYSTAL_STRAIN,
            }.get(self.stage_mode, WorkflowStep.STRUCTURAL_PHASE_PLAN)
            self.workflow_state.mark_completed(step)
            self._register_stage_results(result)
            self.refresh_stage()
            return
        figures = [
            FigureResult(name, image, image_kind="rgb" if np.asarray(image).ndim == 3 else "intensity")
            for name, image in result.images.items()
        ]
        self.workspace.set_results(figures)
        fraction = ", ".join(f"{k}={v:.1%}" for k, v in result.phase_fraction.items())
        warning = " ".join(result.warnings)
        self._notify(
            f"{self.pending_name} complete in {result.elapsed_seconds:.2f}s. Phase fraction: {fraction} {warning}".strip(),
            "warning" if result.warnings else "success",
        )
        self.log_panel.process_finished(self.pending_name, self.status_label.text())
        self.workflow_state.mark_completed(WorkflowStep.STRUCTURAL_PHASE_MATCH)
        if isinstance(self.service, CrystalAnalysisService):
            self.workflow_state.mark_completed(WorkflowStep.CRYSTAL_PHASE)
        self._register_results(result)
        self.phase_result_ready.emit(result)
        self.refresh_stage()

    def _register_results(self, result: PhaseMatchResult) -> None:
        if self.result_registry is None:
            return
        for name, image in result.images.items():
            self.result_registry.register(
                name, "phase", image, ("npy", "png", "tiff"), self.params_snapshot()
            )

    def _register_stage_results(self, result: CrystalAnalysisStageResult) -> None:
        if self.result_registry is None:
            return
        for name, image in result.images.items():
            self.result_registry.register(
                name, "crystal", image, ("npy", "png", "tiff"), self.params_snapshot()
            )

    def _failed(self, message: str) -> None:
        self._notify(f"Failed: {message}", "error")
        self.log_panel.process_failed(self.pending_name or "Phase Mapping", message)
        QMessageBox.warning(self, "Phase Mapping", message)

    def _handle_error(self, message: str) -> None:
        self._failed(message)

    def set_cuda_enabled(self, enabled: bool) -> None:
        self.cuda_enabled = enabled

    def _plan_params(self) -> PhasePlanParams:
        return PhasePlanParams(
            accelerating_voltage=self.voltage.value(), k_max=self.k_max.value(),
            angle_step_zone_axis=self.zone_step.value(), angle_step_in_plane=self.plane_step.value(),
            corr_kernel_size=self.corr_kernel_size.value(), sigma_excitation_error=self.sigma.value(),
            mode=self.mode.currentText(),
            fiber_axis=(self.fiber_x.value(), self.fiber_y.value(), self.fiber_z.value()),
            fiber_angles=(self.fiber_start.value(), self.fiber_end.value()),
            symmetry_order=int(self.symmetry_order.value()), cuda=self.cuda_enabled,
        )

    def _sync_run_config(self) -> None:
        if isinstance(self.service, CrystalAnalysisService):
            self.service.set_run_config(
                CrystalAnalysisRunConfig(
                    mode=self.run_mode.currentText(),
                    roi_size=int(self.roi_size.value()),
                )
            )

    def _crystal_service(self) -> CrystalAnalysisService:
        if not isinstance(self.service, CrystalAnalysisService):
            raise PhaseMappingServiceError("Crystal Analysis service is not available for this page.")
        return self.service

    def _match_params(self) -> PhaseMatchParams:
        return PhaseMatchParams(
            int(self.match_matches.value()), self.match_min_angle.value(),
            int(self.match_min_peaks.value()), self.match_inversion.isChecked(),
            self.corr_normalize.isChecked(), self.low_confidence.value(),
        )

    def params_snapshot(self) -> dict[str, object]:
        plan = self._plan_params()
        match = self._match_params()
        return {
            "mode": plan.mode, "accelerating_voltage": plan.accelerating_voltage, "k_max": plan.k_max,
            "analysis_mode": self.run_mode.currentText(),
            "roi_size": int(self.roi_size.value()),
            "angle_step_zone_axis": plan.angle_step_zone_axis,
            "angle_step_in_plane": plan.angle_step_in_plane,
            "corr_kernel_size": plan.corr_kernel_size,
            "sigma_excitation_error": plan.sigma_excitation_error,
            "fiber_axis": list(plan.fiber_axis), "fiber_angles": list(plan.fiber_angles),
            "symmetry_order": plan.symmetry_order,
            "match_candidates": match.num_matches_return,
            "min_angle_between_matches_deg": match.min_angle_between_matches_deg,
            "min_number_peaks": match.min_number_peaks,
            "inversion_symmetry": match.inversion_symmetry,
            "corr_normalize": match.corr_normalize,
            "low_confidence_threshold": match.low_confidence_threshold,
            "cuda": plan.cuda,
            "crystal_count": len(self.service.crystals),
        }

    def _watch_parameters(self) -> None:
        for widget in [self.mode]:
            self.workflow_state.watch(widget, WorkflowStep.STRUCTURAL_PHASE_PLAN, "currentTextChanged")
            widget.currentTextChanged.connect(lambda _v: self._invalidate_plan())
        self.run_mode.currentTextChanged.connect(lambda _v: self._sync_run_config())
        self.roi_size.valueChanged.connect(lambda _v: self._sync_run_config())
        for widget in [self.voltage, self.k_max, self.zone_step, self.plane_step, self.corr_kernel_size,
                       self.sigma, self.fiber_x, self.fiber_y, self.fiber_z, self.fiber_start,
                       self.fiber_end, self.symmetry_order]:
            self.workflow_state.watch(widget, WorkflowStep.STRUCTURAL_PHASE_PLAN, "valueChanged")
            widget.valueChanged.connect(lambda _v: self._invalidate_plan())
        for widget in [self.match_matches, self.match_min_angle, self.match_min_peaks, self.low_confidence]:
            self.workflow_state.watch(widget, WorkflowStep.STRUCTURAL_PHASE_MATCH, "valueChanged")
            widget.valueChanged.connect(lambda _v: self._invalidate_result())
        self.match_inversion.toggled.connect(lambda _v: self._invalidate_result())
        self.corr_normalize.toggled.connect(lambda _v: self._invalidate_result())
        self.crystal_list.itemChanged.connect(self._crystal_check_changed)

    def _crystal_check_changed(self, item: QListWidgetItem) -> None:
        row = self.crystal_list.row(item)
        if 0 <= row < len(self.service.crystals):
            self.service.set_crystal_enabled(row, item.checkState() == Qt.Checked)
            self._invalidate_plan()

    def _invalidate_plan(self) -> None:
        self.service.invalidate_plan()
        self._notify("Plan parameters changed; recreate the multi-phase plan.", "warning")
        self.refresh_stage()

    def _invalidate_result(self) -> None:
        self.service.invalidate_result()
        self._notify("Match parameters changed; run phase matching again.", "warning")
        self.refresh_stage()

    def _refresh_stale_status(self) -> None:
        step = self.STAGE_STEPS.get(self.stage_mode)
        if step is not None and self.workflow_state.is_stale(step):
            self._notify(STALE_RESULTS_MESSAGE, "warning")
        self.refresh_stage()

    def _notify(self, message: str, level: str = "info") -> None:
        self.status_label.setText(message)

    def _group(self, title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        for label, widget in rows:
            form.addRow(label, widget)
        return group

    def _row(self, *widgets: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return container

    def _float(self, minimum: float, maximum: float, value: float, decimals: int) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals)

    def _int(self, minimum: int, maximum: int, value: int) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, integer=True)
