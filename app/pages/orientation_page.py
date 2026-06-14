from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.services.orientation_service import (
    ManualCrystalParams, OrientationMapParams, OrientationPlanParams,
    OrientationService, OrientationStageResult, SinglePatternMatchParams,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
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
                result = self.operation()
            stream.flush()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class OrientationPage(QWidget):
    STAGE_STEPS = {
        "plan": WorkflowStep.ORIENTATION_PLAN,
        "setup": WorkflowStep.ORIENTATION_REVIEW,
        "review": WorkflowStep.ORIENTATION_REVIEW,
        "map": WorkflowStep.ORIENTATION_MATCH,
    }

    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
        service: OrientationService | None = None,
        stage_mode: str = "plan",
        workspace: AdaptiveImageWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = service or OrientationService()
        self.stage_mode = stage_mode
        self.cuda_enabled = False
        self.worker_thread: QThread | None = None
        self.worker: OrientationWorker | None = None
        self.pending_name = ""
        self.review_position_target: OrientationPage | None = None
        self.workspace = workspace or AdaptiveImageWorkspace()
        self.status_label = QLabel("Orientation workflow ready.")
        self._create_controls()
        self._build_layout()
        self._configure_stage()
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)

    def _create_controls(self) -> None:
        self.crystal_source = QComboBox()
        self.crystal_source.addItems(["CIF", "Manual lattice"])
        self.crystal_label = QLabel("No crystal loaded")
        self.crystal_label.setWordWrap(True)
        self.load_cif_button = QPushButton("Load Crystal CIF")
        self.load_cif_button.clicked.connect(self.load_cif)
        self.lattice_type = QComboBox()
        self.lattice_type.addItems(["cubic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"])
        self.lattice_parameters = QLineEdit("4.08")
        self.space_group = QLineEdit()
        self.atom_table = QTableWidget(1, 4)
        self.atom_table.setHorizontalHeaderLabels(["Element", "x", "y", "z"])
        for column, value in enumerate(("Au", "0", "0", "0")):
            self.atom_table.setItem(0, column, QTableWidgetItem(value))
        self.add_atom_button = QPushButton("Add Atom Row")
        self.add_atom_button.clicked.connect(lambda: self.atom_table.insertRow(self.atom_table.rowCount()))
        self.manual_button = QPushButton("Create Manual Crystal")
        self.manual_button.clicked.connect(self.create_manual_crystal)

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
        self.plan_button = QPushButton("Create Orientation Plan")
        self.plan_button.clicked.connect(lambda: self._start("Orientation Plan", lambda: self.service.create_plan_stage(self._plan_params())))

        self.scan_x = self._int(0, 100000, 0)
        self.scan_y = self._int(0, 100000, 0)
        self.review_matches = self._int(1, 20, 3)
        self.review_min_angle = self._float(0, 180, 5, 2)
        self.review_min_peaks = self._int(1, 1000, 3)
        self.review_inversion = QCheckBox("Use inversion symmetry")
        self.review_inversion.setChecked(True)
        self.simulation_sigma = self._float(0.001, 10, 0.03, 4)
        self.review_button = QPushButton("Run Single-Pattern Review")
        self.review_button.clicked.connect(lambda: self._start(
            "Single-Pattern Review",
            lambda: self.service.review_single_pattern(self.braggvectors_provider(), self._review_params()),
        ))
        self.accept_review_button = QPushButton("Accept Single-Pattern Review")
        self.accept_review_button.clicked.connect(self.accept_review)

        self.map_matches = self._int(1, 20, 2)
        self.map_min_angle = self._float(0, 180, 5, 2)
        self.map_min_peaks = self._int(1, 1000, 3)
        self.map_inversion = QCheckBox("Use inversion symmetry")
        self.map_inversion.setChecked(True)
        self.corr_normalize = QCheckBox("Normalize correlation")
        self.corr_normalize.setChecked(True)
        self.low_confidence = self._float(-1000, 1000, 0.1, 4)
        self.map_button = QPushButton("Run Full Orientation Map")
        self.map_button.clicked.connect(lambda: self._start(
            "Full Orientation Map",
            lambda: self.service.match_map(self.braggvectors_provider(), self._map_params()),
        ))

    def _build_layout(self) -> None:
        self.groups: dict[str, QGroupBox] = {}
        self.groups["crystal"] = self._group("Crystal Setup", [
            ("Source", self.crystal_source), ("Crystal", self.crystal_label), ("", self.load_cif_button),
            ("Lattice type", self.lattice_type), ("Lattice parameters", self.lattice_parameters),
            ("Space group (optional)", self.space_group), ("Atoms / fractional positions", self.atom_table),
            ("", self.add_atom_button), ("", self.manual_button),
        ])
        self.groups["plan"] = self._group("Orientation Search Plan", [
            ("Mode", self.mode), ("Accelerating voltage", self.voltage), ("k_max", self.k_max),
            ("Zone-axis step", self.zone_step), ("In-plane step", self.plane_step),
            ("Correlation kernel", self.corr_kernel_size), ("Excitation sigma", self.sigma),
            ("Fiber axis x", self.fiber_x), ("Fiber axis y", self.fiber_y), ("Fiber axis z", self.fiber_z),
            ("Fiber angle start", self.fiber_start), ("Fiber angle end", self.fiber_end),
            ("Fiber symmetry order", self.symmetry_order), ("", self.plan_button),
        ])
        self.groups["review"] = self._group("Single-Pattern Match Review", [
            ("Scan x", self.scan_x), ("Scan y", self.scan_y), ("Candidates", self.review_matches),
            ("Minimum candidate angle", self.review_min_angle), ("Minimum peaks", self.review_min_peaks),
            ("Simulation sigma", self.simulation_sigma), ("", self.review_inversion),
            ("", self.review_button), ("", self.accept_review_button),
        ])
        self.groups["map"] = self._group("Full Orientation Map & Quality Review", [
            ("Candidates", self.map_matches), ("Minimum candidate angle", self.map_min_angle),
            ("Minimum peaks", self.map_min_peaks), ("Low-confidence threshold", self.low_confidence),
            ("", self.map_inversion), ("", self.corr_normalize), ("", self.map_button),
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
            "plan": {"crystal", "plan", "review"},
            "setup": {"crystal", "plan", "review"},
            "review": {"review"},
            "map": {"map"},
        }[self.stage_mode]
        for name, group in self.groups.items():
            group.setVisible(name in visible)
        self.refresh_stage()

    def refresh_stage(self, *_args) -> None:
        ctx = self.service.context
        self.crystal_label.setText(ctx.crystal_summary)
        self.review_button.setEnabled(ctx.plan_result is not None)
        self.accept_review_button.setEnabled(ctx.single_result is not None)
        self.map_button.setEnabled(ctx.single_review_accepted)

    def set_review_position_target(self, page: "OrientationPage") -> None:
        self.review_position_target = page

    def load_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load crystal structure", "", "CIF files (*.cif)")
        if not path:
            return
        try:
            self.service.load_crystal(path)
            self.crystal_label.setText(self.service.context.crystal_summary)
            self.workflow_state.parameters_updated(WorkflowStep.ORIENTATION_PLAN)
            self._notify("Crystal loaded. Create the orientation plan.", "success")
        except Exception as exc:
            self._failed(str(exc))

    def create_manual_crystal(self) -> None:
        try:
            summary = self.service.create_manual_crystal(self._manual_params())
            self.crystal_label.setText(summary)
            self.workflow_state.parameters_updated(WorkflowStep.ORIENTATION_PLAN)
            self._notify("Manual crystal created. Create the orientation plan.", "success")
        except Exception as exc:
            self._failed(str(exc))

    def accept_review(self) -> None:
        try:
            self.service.accept_single_review()
        except Exception as exc:
            self._failed(str(exc))
            return
        self._notify("Single-pattern review accepted. Full mapping is enabled.", "success")
        self.workflow_state.mark_completed(WorkflowStep.ORIENTATION_REVIEW_ACCEPT)
        self.refresh_stage()

    def _start(self, name: str, operation) -> None:
        if name == "Full Orientation Map":
            self.workspace.lock_auto_layout()
        self.pending_name = name
        self._notify(f"Running {name}...", "info")
        self.log_panel.process_started(name)
        self.log_panel.process_snapshot(ProcessSnapshot(step=name, parameters=self.params_snapshot()))
        self.worker_thread = QThread()
        self.worker = OrientationWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.progress.connect(self.log_panel.process_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _finished(self, result: OrientationStageResult) -> None:
        figures = [
            FigureResult(name, image, image_kind="rgb" if np.asarray(image).ndim == 3 else "intensity")
            for name, image in result.images.items()
        ]
        self.workspace.set_results(figures)
        detail = ", ".join(f"{key}={value}" for key, value in result.metrics.items())
        warning = " ".join(result.warnings)
        self._notify(
            f"{self.pending_name} complete in {result.elapsed_seconds:.2f}s. {detail} {warning}".strip(),
            "warning" if result.warnings else "success",
        )
        self.log_panel.process_finished(self.pending_name, self.status_label.text())
        step = (
            WorkflowStep.ORIENTATION_PLAN
            if self.pending_name == "Orientation Plan"
            else self.STAGE_STEPS[self.stage_mode]
        )
        self.workflow_state.mark_completed(step)
        if self.stage_mode == "map":
            self._register_map_results(result)
            self._connect_map_clicks()
        self.refresh_stage()

    def _register_map_results(self, result: OrientationStageResult) -> None:
        if self.result_registry is None:
            return
        for name, image in result.images.items():
            self.result_registry.register(
                name, "orientation", image, ("npy", "png", "tiff"), self.params_snapshot()
            )

    def _connect_map_clicks(self) -> None:
        if self.review_position_target is None:
            return
        for panel in self.workspace.panels:
            viewer = panel.viewer
            signal = getattr(viewer, "image_clicked", None)
            if signal is None or bool(viewer.property("orientationMapClickConnected")):
                continue
            signal.connect(self._map_clicked)
            viewer.setProperty("orientationMapClickConnected", True)

    def _map_clicked(self, x: int, y: int) -> None:
        target = self.review_position_target
        if target is None:
            return
        target.scan_x.setValue(x)
        target.scan_y.setValue(y)
        target._notify(f"Selected map position ({x}, {y}); run single-pattern review.", "info")
        self.workflow_state.parameters_updated(WorkflowStep.ORIENTATION_REVIEW)

    def _failed(self, message: str) -> None:
        self._notify(f"Failed: {message}", "error")
        self.log_panel.process_failed(self.pending_name or "Orientation", message)
        QMessageBox.warning(self, "Orientation", message)

    def _clear_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None

    def set_cuda_enabled(self, enabled: bool) -> None:
        self.cuda_enabled = enabled

    def _manual_params(self) -> ManualCrystalParams:
        lattice = tuple(float(value.strip()) for value in self.lattice_parameters.text().split(",") if value.strip())
        elements: list[str] = []
        positions: list[tuple[float, float, float]] = []
        for row in range(self.atom_table.rowCount()):
            values = [
                self.atom_table.item(row, column).text().strip()
                if self.atom_table.item(row, column) is not None else ""
                for column in range(4)
            ]
            if not any(values):
                continue
            if not all(values):
                raise ValueError(f"Atom row {row + 1} is incomplete.")
            elements.append(values[0])
            positions.append(tuple(float(value) for value in values[1:4]))
        space_group = self.space_group.text().strip() or None
        return ManualCrystalParams(self.lattice_type.currentText(), lattice, tuple(elements), tuple(positions), space_group)

    def _plan_params(self) -> OrientationPlanParams:
        return OrientationPlanParams(
            accelerating_voltage=self.voltage.value(), k_max=self.k_max.value(),
            angle_step_zone_axis=self.zone_step.value(), angle_step_in_plane=self.plane_step.value(),
            corr_kernel_size=self.corr_kernel_size.value(), sigma_excitation_error=self.sigma.value(),
            mode=self.mode.currentText(), fiber_axis=(self.fiber_x.value(), self.fiber_y.value(), self.fiber_z.value()),
            fiber_angles=(self.fiber_start.value(), self.fiber_end.value()),
            symmetry_order=int(self.symmetry_order.value()), cuda=self.cuda_enabled,
        )

    def _review_params(self) -> SinglePatternMatchParams:
        return SinglePatternMatchParams(
            int(self.scan_x.value()), int(self.scan_y.value()), int(self.review_matches.value()),
            self.review_min_angle.value(), int(self.review_min_peaks.value()),
            self.review_inversion.isChecked(), self.simulation_sigma.value(),
        )

    def _map_params(self) -> OrientationMapParams:
        return OrientationMapParams(
            int(self.map_matches.value()), self.map_min_angle.value(), int(self.map_min_peaks.value()),
            self.map_inversion.isChecked(), self.corr_normalize.isChecked(), self.low_confidence.value(),
        )

    def params_snapshot(self) -> dict[str, object]:
        plan, review, mapping = self._plan_params(), self._review_params(), self._map_params()
        return {
            "crystal_source": self.crystal_source.currentText(), "lattice_type": self.lattice_type.currentText(),
            "lattice_parameters": self.lattice_parameters.text(), "space_group": self.space_group.text(),
            "mode": plan.mode, "accelerating_voltage": plan.accelerating_voltage, "k_max": plan.k_max,
            "angle_step_zone_axis": plan.angle_step_zone_axis, "angle_step_in_plane": plan.angle_step_in_plane,
            "corr_kernel_size": plan.corr_kernel_size, "sigma_excitation_error": plan.sigma_excitation_error,
            "fiber_axis": list(plan.fiber_axis), "fiber_angles": list(plan.fiber_angles),
            "symmetry_order": plan.symmetry_order, "scan_x": review.scan_x, "scan_y": review.scan_y,
            "review_matches": review.num_matches_return, "map_matches": mapping.num_matches_return,
            "min_angle_between_matches_deg": mapping.min_angle_between_matches_deg,
            "min_number_peaks": mapping.min_number_peaks, "inversion_symmetry": mapping.inversion_symmetry,
            "low_confidence_threshold": mapping.low_confidence_threshold, "cuda": plan.cuda,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        text_controls = [("lattice_parameters", self.lattice_parameters), ("space_group", self.space_group)]
        for key, control in text_controls:
            if key in params:
                control.setText(str(params[key]))
        combos = [("crystal_source", self.crystal_source), ("lattice_type", self.lattice_type), ("mode", self.mode)]
        for key, combo in combos:
            if key in params:
                combo.setCurrentText(str(params[key]))
        for key, control in [
            ("accelerating_voltage", self.voltage), ("k_max", self.k_max),
            ("angle_step_zone_axis", self.zone_step), ("angle_step_in_plane", self.plane_step),
            ("corr_kernel_size", self.corr_kernel_size), ("sigma_excitation_error", self.sigma),
            ("symmetry_order", self.symmetry_order), ("scan_x", self.scan_x), ("scan_y", self.scan_y),
            ("review_matches", self.review_matches), ("map_matches", self.map_matches),
            ("min_angle_between_matches_deg", self.map_min_angle), ("min_number_peaks", self.map_min_peaks),
            ("low_confidence_threshold", self.low_confidence),
        ]:
            if key in params:
                control.setValue(float(params[key]))
        for key, controls in [("fiber_axis", (self.fiber_x, self.fiber_y, self.fiber_z)), ("fiber_angles", (self.fiber_start, self.fiber_end))]:
            if key in params:
                for control, value in zip(controls, params[key]):
                    control.setValue(float(value))
        if "inversion_symmetry" in params:
            self.map_inversion.setChecked(bool(params["inversion_symmetry"]))

    def _watch_parameters(self) -> None:
        for widget in [self.mode, self.lattice_type, self.crystal_source]:
            self.workflow_state.watch(widget, WorkflowStep.ORIENTATION_PLAN, "currentTextChanged")
            widget.currentTextChanged.connect(lambda _value: self._invalidate_plan())
        for widget in [self.lattice_parameters, self.space_group]:
            self.workflow_state.watch(widget, WorkflowStep.ORIENTATION_PLAN, "textChanged")
            widget.textChanged.connect(lambda _value: self._invalidate_plan())
        self.atom_table.cellChanged.connect(lambda *_args: self._invalidate_plan())
        for widget in [self.voltage, self.k_max, self.zone_step, self.plane_step, self.corr_kernel_size, self.sigma,
                       self.fiber_x, self.fiber_y, self.fiber_z, self.fiber_start, self.fiber_end, self.symmetry_order]:
            self.workflow_state.watch(widget, WorkflowStep.ORIENTATION_PLAN, "valueChanged")
            widget.valueChanged.connect(lambda _value: self._invalidate_plan())
        for widget in [self.scan_x, self.scan_y, self.review_matches, self.review_min_angle, self.review_min_peaks, self.simulation_sigma]:
            self.workflow_state.watch(widget, WorkflowStep.ORIENTATION_REVIEW, "valueChanged")
            widget.valueChanged.connect(lambda _value: self._invalidate_review())
        self.review_inversion.toggled.connect(lambda _value: self._invalidate_review())
        for widget in [self.map_matches, self.map_min_angle, self.map_min_peaks, self.low_confidence]:
            self.workflow_state.watch(widget, WorkflowStep.ORIENTATION_MATCH, "valueChanged")
            widget.valueChanged.connect(lambda _value: self._invalidate_map())
        self.map_inversion.toggled.connect(lambda _value: self._invalidate_map())
        self.corr_normalize.toggled.connect(lambda _value: self._invalidate_map())

    def _invalidate_plan(self) -> None:
        self.service.invalidate_plan()
        self._notify("Orientation plan parameters changed; recreate the plan and review.", "warning")
        self.refresh_stage()

    def _invalidate_review(self) -> None:
        self.service.invalidate_review()
        self._notify("Single-pattern review parameters changed; review and accept again.", "warning")
        self.refresh_stage()

    def _invalidate_map(self) -> None:
        self.service.invalidate_map()
        self._notify("Orientation map parameters changed; run the full map again.", "warning")
        self.refresh_stage()

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.is_stale(self.STAGE_STEPS[self.stage_mode]):
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

    def _float(self, minimum: float, maximum: float, value: float, decimals: int) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals)

    def _int(self, minimum: int, maximum: int, value: int) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, integer=True)
