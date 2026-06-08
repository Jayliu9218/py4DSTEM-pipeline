from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.orientation_service import (
    OrientationMatchParams,
    OrientationPlanParams,
    OrientationResult,
    OrientationService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel, ProcessSnapshot
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
    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = OrientationService()
        self.cuda_enabled = False
        self.worker_thread: QThread | None = None
        self.worker: OrientationWorker | None = None
        self.current_process_step = WorkflowStep.ORIENTATION_PLAN
        self.crystal_label = QLabel("No CIF loaded")
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.voltage = self._spin(1000, 1000000, 300000)
        self.k_max = self._spin(0.1, 10, 1.5)
        self.zone_step = self._spin(0.1, 30, 2)
        self.plane_step = self._spin(0.1, 30, 2)
        self.corr_kernel_size = self._spin(0.001, 10, 0.08, decimals=4)
        self.sigma_excitation_error = self._spin(0.001, 10, 0.02, decimals=4)
        self.num_matches = QSpinBox()
        self.num_matches.setRange(1, 20)
        self.num_matches.setValue(1)
        self.min_match_angle = self._spin(0, 180, 5)
        self.min_match_peaks = QSpinBox()
        self.min_match_peaks.setRange(1, 1000)
        self.min_match_peaks.setValue(3)
        self.inversion_symmetry = QCheckBox("Use inversion symmetry")
        self.inversion_symmetry.setChecked(True)
        self.load_button = QPushButton("Load Crystal CIF")
        self.plan_button = QPushButton("Create Orientation Plan")
        self.match_button = QPushButton("Match and Show Orientation Map")
        self.buttons = [self.load_button, self.plan_button, self.match_button]
        self.match_progress = QProgressBar()
        self.match_progress.setRange(0, 0)
        self.match_progress.setFormat("Matching orientation map...")
        self.match_progress.setTextVisible(True)
        self.match_progress.setVisible(False)
        self.viewer_tabs = QTabWidget()
        self.viewers: dict[str, ImageViewer] = {}
        for name in [
            "Orientation RGB",
            "Correlation Score",
            "Reliability",
            "Peak Count",
            "Ambiguity",
        ]:
            viewer = ImageViewer("color" if name == "Orientation RGB" else "intensity")
            self.viewers[name] = viewer
            self.viewer_tabs.addTab(viewer, name)
        self.load_button.clicked.connect(self.load_cif)
        self.plan_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.create_plan(self._params()),
                WorkflowStep.ORIENTATION_PLAN,
            )
        )
        self.match_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.match(self.braggvectors_provider(), self._match_params()),
                WorkflowStep.ORIENTATION_MATCH,
            )
        )
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        crystal_group = QGroupBox("1 Crystal Reference")
        crystal_layout = QVBoxLayout(crystal_group)
        crystal_form = QFormLayout()
        crystal_form.addRow("Crystal", self.crystal_label)
        crystal_layout.addLayout(crystal_form)
        crystal_layout.addWidget(self.load_button)

        plan_group = QGroupBox("2 Orientation Plan Parameters")
        plan_layout = QFormLayout(plan_group)
        plan_layout.addRow("accelerating voltage", self.voltage)
        plan_layout.addRow("k_max", self.k_max)
        plan_layout.addRow("zone-axis angle step", self.zone_step)
        plan_layout.addRow("in-plane angle step", self.plane_step)
        plan_layout.addRow("correlation kernel size", self.corr_kernel_size)
        plan_layout.addRow("excitation-error sigma", self.sigma_excitation_error)
        plan_layout.addRow("", self.plan_button)

        match_group = QGroupBox("3 Orientation Matching")
        match_layout = QFormLayout(match_group)
        match_layout.addRow("matches to return", self.num_matches)
        match_layout.addRow("minimum match angle", self.min_match_angle)
        match_layout.addRow("minimum matched peaks", self.min_match_peaks)
        match_layout.addRow("", self.inversion_symmetry)
        match_layout.addRow("", self.match_button)
        match_layout.addRow("", self.match_progress)

        status_group = QGroupBox("4 Results")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.status_label)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        for button in self.buttons:
            button.setMinimumHeight(30)
        for group in [crystal_group, plan_group, match_group, status_group]:
            left_layout.addWidget(group)
        left_layout.addStretch(1)
        self.controls_panel = left
        layout = QHBoxLayout(self)
        layout.addWidget(self.viewer_tabs)

    def load_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load crystal structure", "", "CIF files (*.cif)")
        if not path:
            return
        try:
            self.crystal_label.setText(self.service.load_crystal(path))
            self.status_label.setText("Crystal loaded")
            self.workflow_state.parameters_updated(WorkflowStep.ORIENTATION_PLAN)
        except Exception as exc:
            QMessageBox.warning(self, "Orientation", str(exc))

    def set_cuda_enabled(self, enabled: bool) -> None:
        self.cuda_enabled = enabled

    def _params(self) -> OrientationPlanParams:
        return OrientationPlanParams(
            accelerating_voltage=self.voltage.value(),
            k_max=self.k_max.value(),
            angle_step_zone_axis=self.zone_step.value(),
            angle_step_in_plane=self.plane_step.value(),
            corr_kernel_size=self.corr_kernel_size.value(),
            sigma_excitation_error=self.sigma_excitation_error.value(),
            cuda=self.cuda_enabled,
        )

    def _match_params(self) -> OrientationMatchParams:
        return OrientationMatchParams(
            num_matches_return=self.num_matches.value(),
            min_angle_between_matches_deg=self.min_match_angle.value(),
            min_number_peaks=self.min_match_peaks.value(),
            inversion_symmetry=self.inversion_symmetry.isChecked(),
        )

    def _run(self, operation, process_step: str) -> None:
        if process_step == WorkflowStep.ORIENTATION_MATCH:
            warning = self._calibration_warning()
            if warning:
                self.log_panel.log(f"WARN  {warning}")
                self.status_label.setText(warning)
        for button in self.buttons:
            button.setEnabled(False)
        self.status_label.setText("Running orientation step...")
        self.current_process_step = process_step
        self.match_progress.setVisible(process_step == WorkflowStep.ORIENTATION_MATCH)
        self.log_panel.process_started("Orientation analysis")
        if process_step == WorkflowStep.ORIENTATION_PLAN:
            params = self._params()
            snapshot_params = {
                "voltage": params.accelerating_voltage,
                "k_max": params.k_max,
                "zone_step": params.angle_step_zone_axis,
                "plane_step": params.angle_step_in_plane,
                "CUDA": params.cuda,
            }
        else:
            params = self._match_params()
            snapshot_params = {
                "matches": params.num_matches_return,
                "min_angle": params.min_angle_between_matches_deg,
                "min_peaks": params.min_number_peaks,
                "inversion_symmetry": params.inversion_symmetry,
            }
        self.log_panel.process_snapshot(
            ProcessSnapshot(step="Orientation analysis", parameters=snapshot_params)
        )
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

    def _calibration_warning(self) -> str:
        braggvectors = self.braggvectors_provider()
        calstate = getattr(braggvectors, "calstate", {}) if braggvectors is not None else {}
        missing = [
            label
            for name, label in [
                ("center", "origin"),
                ("ellipse", "ellipse"),
                ("pixel", "pixel"),
                ("rotate", "rotation"),
            ]
            if not bool(getattr(calstate, "get", lambda _name, _default=False: False)(name, False))
        ]
        if not missing:
            return ""
        return (
            "Calibration is incomplete; orientation will continue, but accuracy may be lower. "
            f"Missing/applied-off corrections: {', '.join(missing)}."
        )

    def _finished(self, result) -> None:
        if isinstance(result, OrientationResult):
            for name, viewer in self.viewers.items():
                image = result.quality.maps.get(name)
                if image is None:
                    viewer.clear(f"{name} is not available for this result.")
                else:
                    viewer.set_image(image)
                    if self.result_registry is not None:
                        self.result_registry.register(
                            name,
                            "orientation",
                            image,
                            ("npy", "png", "tiff"),
                            self.params_snapshot(),
                        )
            self.status_label.setText(f"Orientation map ready in {result.elapsed_seconds:.2f} s")
            if result.quality.warnings:
                self.status_label.setText(
                    self.status_label.text() + " | " + " ".join(result.quality.warnings)
                )
        else:
            self.status_label.setText(f"Orientation plan ready in {float(result):.2f} s")
        self.log_panel.log(self.status_label.text())
        self.log_panel.process_finished("Orientation analysis", self.status_label.text())
        self.workflow_state.mark_completed(self.current_process_step)

    def _failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Orientation failed: {message}")
        self.log_panel.process_failed("Orientation analysis", message)
        QMessageBox.warning(self, "Orientation", message)

    def _clear(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.match_progress.setVisible(False)
        for button in self.buttons:
            button.setEnabled(True)

    def _spin(self, minimum, maximum, value, decimals=2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _watch_parameters(self) -> None:
        for spin in [
            self.voltage,
            self.k_max,
            self.zone_step,
            self.plane_step,
            self.corr_kernel_size,
            self.sigma_excitation_error,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.ORIENTATION_PLAN, "valueChanged")
        for spin in [self.num_matches, self.min_match_angle, self.min_match_peaks]:
            self.workflow_state.watch(spin, WorkflowStep.ORIENTATION_MATCH, "valueChanged")
        self.workflow_state.watch(
            self.inversion_symmetry, WorkflowStep.ORIENTATION_MATCH, "toggled"
        )

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale(
            [WorkflowStep.ORIENTATION_PLAN, WorkflowStep.ORIENTATION_MATCH]
        ):
            self.status_label.setText(STALE_RESULTS_MESSAGE)

    def params_snapshot(self) -> dict[str, object]:
        plan = self._params()
        match = self._match_params()
        return {
            "crystal": self.crystal_label.text(),
            "accelerating_voltage": plan.accelerating_voltage,
            "k_max": plan.k_max,
            "angle_step_zone_axis": plan.angle_step_zone_axis,
            "angle_step_in_plane": plan.angle_step_in_plane,
            "corr_kernel_size": plan.corr_kernel_size,
            "sigma_excitation_error": plan.sigma_excitation_error,
            "num_matches_return": match.num_matches_return,
            "min_angle_between_matches_deg": match.min_angle_between_matches_deg,
            "min_number_peaks": match.min_number_peaks,
            "inversion_symmetry": match.inversion_symmetry,
            "cuda": plan.cuda,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        for key, spin in [
            ("accelerating_voltage", self.voltage),
            ("k_max", self.k_max),
            ("angle_step_zone_axis", self.zone_step),
            ("angle_step_in_plane", self.plane_step),
            ("corr_kernel_size", self.corr_kernel_size),
            ("sigma_excitation_error", self.sigma_excitation_error),
            ("min_angle_between_matches_deg", self.min_match_angle),
        ]:
            if key in params:
                spin.setValue(float(params[key]))
        for key, spin in [
            ("num_matches_return", self.num_matches),
            ("min_number_peaks", self.min_match_peaks),
        ]:
            if key in params:
                spin.setValue(int(params[key]))
        if "inversion_symmetry" in params:
            self.inversion_symmetry.setChecked(bool(params["inversion_symmetry"]))
