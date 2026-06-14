from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import (
    BraggStrainService,
    CalibrationActionResult,
    CrystalPixelParams,
    OriginCalibrationParams,
    QRComparisonParams,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.theme import Theme
from app.widgets.image_viewer import ImageViewer
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class CalibrationPage(QWidget, WorkerRunner):
    def __init__(
        self,
        datacube_provider: Callable[[], object | None],
        braggvectors_provider: Callable[[], object | None],
        ellipse_braggvectors_provider: Callable[[], object | None] | None,
        transfer_targets_provider: Callable[[], list[tuple[str, object]]] | None,
        rotation_reference_provider: Callable[[], object | None] | None,
        service: BraggStrainService,
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.braggvectors_provider = braggvectors_provider
        self.ellipse_braggvectors_provider = ellipse_braggvectors_provider
        self.transfer_targets_provider = transfer_targets_provider
        self.rotation_reference_provider = rotation_reference_provider
        self.service = service
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self._init_worker_runner()
        self.current_process_name = "Calibration step"
        self.current_process_step = WorkflowStep.CALIBRATION_APPLY

        self.source_label = QLabel("-")
        self.origin_label = QLabel("-")
        self.ellipse_label = QLabel("-")
        self.pixel_label = QLabel("-")
        self.rotate_label = QLabel("-")
        self.decision_labels = {
            "origin": QLabel("-"),
            "ellipse": QLabel("-"),
            "pixel": QLabel("-"),
            "rotate": QLabel("-"),
        }
        self.complete_label = QLabel("-")
        self.applied_label = QLabel("none")
        self.origin_measurement_label = QLabel("-")
        self.origin_measurement_label.setWordWrap(True)
        self.ellipse_measurement_label = QLabel("-")
        self.ellipse_measurement_label.setWordWrap(True)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.analysis_target = QComboBox()
        self.analysis_target.addItems(["Preview", "ACOM", "Strain", "DPC"])
        self.origin_center_x = self._float_input(-100000, 100000, 0, unit="px")
        self.origin_center_y = self._float_input(-100000, 100000, 0, unit="px")
        self.origin_use_guess = QCheckBox("Use center guess")
        self.origin_score = QComboBox()
        self.origin_score.addItems(["automatic", "distance", "intensity", "intensity weighted distance"])
        self.origin_find_center = QComboBox()
        self.origin_find_center.addItems(["max", "CoM"])
        self.origin_fit_function = QComboBox()
        self.origin_fit_function.addItems(["plane", "parabola", "bezier_two"])
        self.origin_robust = QCheckBox("Enable robust fit")
        self.origin_robust_steps = self._int_input(1, 20, 3)
        self.origin_robust_threshold = self._float_input(0.1, 100, 2)
        self.ellipse_center_x = self._float_input(0, 100000, 0, unit="px")
        self.ellipse_center_y = self._float_input(0, 100000, 0, unit="px")
        self.ellipse_inner = self._float_input(0.1, 100000, 50, unit="px")
        self.ellipse_outer = self._float_input(0.1, 100000, 100, unit="px")
        self.sampling_spin = self._int_input(1, 64, 8, unit="x")
        self.pixel_spin = self._float_input(0.000001, 1000, 0.02, decimals=6, unit="A^-1")
        self.crystal_cif_path = ""
        self.crystal_path_label = QLabel("Manual FCC crystal")
        self.crystal_path_label.setWordWrap(True)
        self.crystal_lattice = self._float_input(0.000001, 1000, 4.08, decimals=5, unit="A")
        self.crystal_atomic_number = self._int_input(1, 118, 79)
        self.crystal_k_max = self._float_input(0.000001, 1000, 1.5, decimals=4, unit="A^-1")
        self.rotation_spin = self._float_input(-360, 360, -83, unit="deg")
        self.rotation_real_direction = self._float_input(-360, 360, 158, unit="deg")
        self.rotation_real_x = self._float_input(-100000, 100000, 59, unit="px")
        self.rotation_real_y = self._float_input(-100000, 100000, 16.5, unit="px")
        self.rotation_q_x = self._float_input(-100000, 100000, 154, unit="px")
        self.rotation_q_y = self._float_input(-100000, 100000, 205, unit="px")
        self.rotation_real_length = self._float_input(0.01, 2, 0.4, decimals=3)
        self.rotation_q_length = self._float_input(0.01, 2, 0.3, decimals=3)
        self.refresh_button = QPushButton("Check Calibration")
        self.origin_button = QPushButton("Measure Origin")
        self.compare_origin_button = QPushButton("Compare Origin Correction")
        self.draw_ellipse_circle_button = QPushButton("Draw Ring Fit ROI")
        self.ellipse_button = QPushButton("Fit Ellipse")
        self.pixel_button = QPushButton("Set Q Pixel Size")
        self.load_cif_button = QPushButton("Load Crystal CIF")
        self.fit_pixel_button = QPushButton("Fit Pixel Size From Crystal Reference")
        self.rotation_button = QPushButton("Set QR Rotation")
        self.apply_origin_button = QPushButton("Apply")
        self.apply_ellipse_button = QPushButton("Accept && Apply Ellipse")
        self.apply_ellipse_button.setEnabled(False)
        self.apply_pixel_button = QPushButton("Apply")
        self.apply_rotation_button = QPushButton("Apply")
        self.transfer_correction = QComboBox()
        self.transfer_correction.addItem("Origin", "origin")
        self.transfer_correction.addItem("Ellipse", "ellipse")
        self.transfer_correction.addItem("Q Pixel Size", "pixel")
        self.transfer_correction.addItem("QR Rotation", "rotate")
        self.transfer_target = QComboBox()
        self.transfer_button = QPushButton("Transfer Calibration")
        self.validate_button = QPushButton("Validate Calibration")
        self.reset_button = QPushButton("Reset Applied Calibration")
        self.buttons = [
            self.refresh_button,
            self.origin_button,
            self.compare_origin_button,
            self.draw_ellipse_circle_button,
            self.ellipse_button,
            self.pixel_button,
            self.load_cif_button,
            self.fit_pixel_button,
            self.rotation_button,
            self.apply_origin_button,
            self.apply_ellipse_button,
            self.apply_pixel_button,
            self.apply_rotation_button,
            self.transfer_button,
            self.validate_button,
            self.reset_button,
        ]
        self.viewers = AdaptiveImageWorkspace()
        self.figure_results: dict[str, FigureResult] = {}

        self.refresh_button.clicked.connect(lambda: self.refresh_status())
        self.origin_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.calibrate_origin(
                    self.braggvectors_provider(), self._origin_params()
                ),
                "Measure and fit origin",
                WorkflowStep.CALIBRATION_ORIGIN,
            )
        )
        self.apply_origin_button.clicked.connect(
            lambda: self._apply_single_correction("center", "Apply origin correction")
        )
        self.compare_origin_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.compare_origin_correction(self.braggvectors_provider()),
                "Compare origin correction",
                WorkflowStep.CALIBRATION_ORIGIN,
            )
        )
        self.draw_ellipse_circle_button.clicked.connect(self.start_ellipse_circle_draw)
        self.ellipse_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.calibrate_ellipse(
                    self.braggvectors_provider(),
                    self.ellipse_inner.value(),
                    self.ellipse_outer.value(),
                    self.sampling_spin.value(),
                    center=(self.ellipse_center_x.value(), self.ellipse_center_y.value()),
                    fit_source=(
                        self.ellipse_braggvectors_provider()
                        if self.ellipse_braggvectors_provider is not None
                        else None
                    ),
                ),
                "Fit ellipticity",
                WorkflowStep.CALIBRATION_ELLIPSE,
            )
        )
        self.apply_ellipse_button.clicked.connect(
            lambda: self._run(
                self.service.accept_pending_ellipse,
                "Accept and apply ellipse correction",
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
        self.load_cif_button.clicked.connect(self._load_crystal_cif)
        self.fit_pixel_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.fit_pixel_size_from_crystal(
                    (
                        self.ellipse_braggvectors_provider()
                        if self.ellipse_braggvectors_provider is not None
                        else self.braggvectors_provider()
                    ),
                    CrystalPixelParams(
                        cif_path=self.crystal_cif_path or None,
                        lattice_parameter=self.crystal_lattice.value(),
                        atomic_number=int(self.crystal_atomic_number.value()),
                        k_max=self.crystal_k_max.value(),
                        initial_pixel_size=self.pixel_spin.value(),
                    ),
                ),
                "Fit Q pixel size from crystal reference",
                WorkflowStep.CALIBRATION_PIXEL,
            )
        )
        self.apply_pixel_button.clicked.connect(
            lambda: self._apply_single_correction("pixel", "Apply pixel-size correction")
        )
        self.rotation_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.set_qr_rotation(
                    self.braggvectors_provider(),
                    self.rotation_spin.value(),
                    reference_image=(
                        self.rotation_reference_provider()
                        if self.rotation_reference_provider is not None
                        else None
                    ),
                    comparison=self._qr_comparison_params(),
                ),
                "Set QR rotation",
                WorkflowStep.CALIBRATION_ROTATION,
            )
        )
        self.apply_rotation_button.clicked.connect(
            lambda: self._apply_single_correction("rotate", "Apply QR rotation correction")
        )
        self.transfer_button.clicked.connect(self._transfer_selected_correction)
        self.validate_button.clicked.connect(
            lambda: self._run(
                lambda: self.service.validate_calibration(self.braggvectors_provider()),
                "Validate calibration",
                WorkflowStep.CALIBRATION_APPLY,
            )
        )
        self.reset_button.clicked.connect(self.reset_calibration)
        self.analysis_target.currentTextChanged.connect(lambda _text: self._refresh_decision_panel())
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()
        self.refresh_transfer_targets()
        self.show_braggvectors_histogram()

    def _load_crystal_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load crystal CIF", "", "CIF files (*.cif)")
        if path:
            self.crystal_cif_path = path
            self.crystal_path_label.setText(path)
            self.workflow_state.parameters_updated(WorkflowStep.CALIBRATION_PIXEL)

    def _build_layout(self) -> None:
        status_group = QGroupBox("Existing Calibration")
        status_layout = QVBoxLayout(status_group)
        status_form = QFormLayout()
        status_form.addRow("Analysis target", self.analysis_target)
        for name, label in [
            ("Origin", self.decision_labels["origin"]),
            ("Ellipse", self.decision_labels["ellipse"]),
            ("Q pixel size", self.decision_labels["pixel"]),
            ("QR rotation", self.decision_labels["rotate"]),
        ]:
            status_form.addRow(name, label)
        for label, widget in [
            ("Source", self.source_label),
            ("origin", self.origin_label),
            ("ellipse", self.ellipse_label),
            ("pixel", self.pixel_label),
            ("rotate", self.rotate_label),
            ("measurements", self.complete_label),
            ("applied", self.applied_label),
        ]:
            status_form.addRow(label, widget)
        status_layout.addLayout(status_form)
        status_layout.addWidget(self.refresh_button)
        status_layout.addWidget(self.reset_button)

        origin_group = QGroupBox("Origin Calibration")
        origin_layout = QVBoxLayout(origin_group)
        origin_layout.addWidget(self.origin_measurement_label)
        origin_form = QFormLayout()
        origin_form.addRow("", self.origin_use_guess)
        origin_form.addRow("center guess x", self.origin_center_x)
        origin_form.addRow("center guess y", self.origin_center_y)
        origin_form.addRow("score method", self.origin_score)
        origin_form.addRow("find center", self.origin_find_center)
        origin_form.addRow("fit function", self.origin_fit_function)
        origin_form.addRow("", self.origin_robust)
        origin_form.addRow("robust steps", self.origin_robust_steps)
        origin_form.addRow("robust threshold", self.origin_robust_threshold)
        origin_layout.addLayout(origin_form)
        origin_layout.addWidget(self.origin_button)
        origin_layout.addWidget(self.compare_origin_button)
        origin_layout.addWidget(self.apply_origin_button)

        ellipse_group = QGroupBox("Ellipse Calibration")
        ellipse_layout = QFormLayout(ellipse_group)
        ellipse_layout.addRow("fit result", self.ellipse_measurement_label)
        ellipse_layout.addRow("fit center x", self.ellipse_center_x)
        ellipse_layout.addRow("fit center y", self.ellipse_center_y)
        ellipse_layout.addRow("inner radius", self.ellipse_inner)
        ellipse_layout.addRow("outer radius", self.ellipse_outer)
        ellipse_layout.addRow("BVM sampling", self.sampling_spin)
        ellipse_layout.addRow("", self.draw_ellipse_circle_button)
        ellipse_layout.addRow("", self.ellipse_button)
        ellipse_layout.addRow("", self.apply_ellipse_button)

        pixel_group = QGroupBox("Q Pixel Size")
        pixel_layout = QFormLayout(pixel_group)
        pixel_layout.addRow("Q pixel size", self.pixel_spin)
        pixel_layout.addRow("", self.pixel_button)
        pixel_layout.addRow("crystal source", self.crystal_path_label)
        pixel_layout.addRow("", self.load_cif_button)
        pixel_layout.addRow("FCC lattice a", self.crystal_lattice)
        pixel_layout.addRow("atomic number", self.crystal_atomic_number)
        pixel_layout.addRow("k max", self.crystal_k_max)
        pixel_layout.addRow("", self.fit_pixel_button)
        pixel_layout.addRow("", self.apply_pixel_button)

        rotation_group = QGroupBox("QR Rotation")
        rotation_layout = QFormLayout(rotation_group)
        rotation_layout.addRow("QR rotation", self.rotation_spin)
        rotation_layout.addRow("R direction", self.rotation_real_direction)
        rotation_layout.addRow("R x", self.rotation_real_x)
        rotation_layout.addRow("R y", self.rotation_real_y)
        rotation_layout.addRow("Q x", self.rotation_q_x)
        rotation_layout.addRow("Q y", self.rotation_q_y)
        rotation_layout.addRow("R length", self.rotation_real_length)
        rotation_layout.addRow("Q length", self.rotation_q_length)
        rotation_layout.addRow("", self.rotation_button)
        rotation_layout.addRow("", self.apply_rotation_button)

        transfer_group = QGroupBox("Transfer")
        transfer_layout = QFormLayout(transfer_group)
        transfer_layout.addRow("correction", self.transfer_correction)
        transfer_layout.addRow("target DataCube", self.transfer_target)
        transfer_layout.addRow("", self.transfer_button)
        self.calibration_forms = {
            "Existing Calibration": status_form,
            "Origin Calibration": origin_form,
            "Ellipse Calibration": ellipse_layout,
            "Q Pixel Size": pixel_layout,
            "QR Rotation": rotation_layout,
            "Transfer": transfer_layout,
        }

        validate_group = QGroupBox("Validate")
        validate_layout = QVBoxLayout(validate_group)
        validate_layout.addWidget(self.validate_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSizeConstraint(QLayout.SetNoConstraint)
        for button in self.buttons:
            button.setMinimumHeight(30)
        for group in [
            status_group,
            origin_group,
            ellipse_group,
            pixel_group,
            rotation_group,
            transfer_group,
            validate_group,
        ]:
            left_layout.addWidget(group)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        left.setMinimumSize(0, 0)
        left.setMaximumWidth(16777215)
        left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.controls_panel = left
        layout = QHBoxLayout(self)
        layout.addWidget(self.viewers)
        status_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        status_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for name, form in self.calibration_forms.items():
            if name == "Existing Calibration":
                continue
            form.setRowWrapPolicy(QFormLayout.DontWrapRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for control in [
            *left.findChildren(NumericLineEdit),
            *left.findChildren(QComboBox),
        ]:
            control.setMinimumWidth(0)
            control.setMaximumWidth(16777215)
            control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for control in left.findChildren(QComboBox):
            control.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(0)
        for label in [
            self.source_label,
            self.origin_label,
            self.ellipse_label,
            self.pixel_label,
            self.rotate_label,
            self.complete_label,
            self.applied_label,
            *self.decision_labels.values(),
        ]:
            label.setWordWrap(True)
            label.setMinimumWidth(0)

    def _origin_params(self) -> OriginCalibrationParams:
        use_guess = self.origin_use_guess.isChecked()
        return OriginCalibrationParams(
            center_guess_x=self.origin_center_x.value() if use_guess else None,
            center_guess_y=self.origin_center_y.value() if use_guess else None,
            score_method=None if self.origin_score.currentText() == "automatic" else self.origin_score.currentText(),
            find_center=self.origin_find_center.currentText(),
            fit_function=self.origin_fit_function.currentText(),
            robust=self.origin_robust.isChecked(),
            robust_steps=self.origin_robust_steps.value(),
            robust_threshold=self.origin_robust_threshold.value(),
        )

    def _qr_comparison_params(self) -> QRComparisonParams:
        return QRComparisonParams(
            real_rotation=self.rotation_real_direction.value(),
            real_position_x=self.rotation_real_x.value(),
            real_position_y=self.rotation_real_y.value(),
            reciprocal_position_x=self.rotation_q_x.value(),
            reciprocal_position_y=self.rotation_q_y.value(),
            real_length_fraction=self.rotation_real_length.value(),
            reciprocal_length_fraction=self.rotation_q_length.value(),
        )

    def refresh_status(self, show_histogram: bool = True) -> None:
        braggvectors = self.braggvectors_provider()
        source = braggvectors if braggvectors is not None else self.datacube_provider()
        status = self.service.calibration_status(source)
        self.source_label.setText("BraggVectors" if braggvectors is not None else "DataCube")
        self.complete_label.setText("yes" if status.complete else "no")
        if braggvectors is not None:
            state = braggvectors.calstate
            enabled = [name for name, value in state.items() if value]
            self.applied_label.setText(", ".join(enabled) if enabled else "none")
        else:
            self.applied_label.setText("none")
            enabled = []
        applied = set(enabled)
        self.origin_label.setText(self._status_text(status.origin, "center" in applied))
        self.ellipse_label.setText(self._status_text(status.ellipse, "ellipse" in applied))
        self.pixel_label.setText(self._status_text(status.pixel, "pixel" in applied))
        self.rotate_label.setText(self._status_text(status.rotate, "rotate" in applied))
        self._style_status_labels(status, set(enabled))
        self._refresh_decision_panel()
        self.refresh_transfer_targets()
        if show_histogram:
            self.show_braggvectors_histogram()

    def reset_calibration(self) -> None:
        self._run(
            lambda: self.service.set_calibration_state(
                self.braggvectors_provider(),
                False,
                False,
                False,
                False,
            ),
            "Reset applied calibration",
            WorkflowStep.CALIBRATION_APPLY,
        )

    def show_braggvectors_histogram(self, make_current: bool = False) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            return
        try:
            image = braggvectors.histogram(mode="raw").data
        except Exception as exc:
            self.log_panel.log(f"Could not display BraggVectors histogram: {exc}")
            return
        self._set_viewer_tab("BraggVectors histogram", image, make_current=make_current)
        self._set_default_ellipse_center(image)

    def _apply_single_correction(self, calstate_name: str, process_name: str) -> None:
        braggvectors = self.braggvectors_provider()
        state = dict(getattr(braggvectors, "calstate", {})) if braggvectors is not None else {}
        next_state = {
            "center": bool(state.get("center", False)),
            "ellipse": bool(state.get("ellipse", False)),
            "pixel": bool(state.get("pixel", False)),
            "rotate": bool(state.get("rotate", False)),
        }
        next_state[calstate_name] = True
        self._run(
            lambda: self.service.set_calibration_state(
                self.braggvectors_provider(),
                next_state["center"],
                next_state["ellipse"],
                next_state["pixel"],
                next_state["rotate"],
            ),
            process_name,
            WorkflowStep.CALIBRATION_APPLY,
        )

    def _transfer_selected_correction(self) -> None:
        if self.transfer_targets_provider is None:
            QMessageBox.information(
                self,
                "Calibration Transfer",
                "Run BraggVectors on at least two DataCubes first.",
            )
            return
        self.refresh_transfer_targets()
        target = self.transfer_target.currentData()
        if target is None:
            QMessageBox.information(
                self,
                "Calibration Transfer",
                "Run BraggVectors on another DataCube first, then choose it as the transfer target.",
            )
            return
        label = self.transfer_correction.currentText()
        correction = str(self.transfer_correction.currentData())
        target_label = self.transfer_target.currentText()
        self._run(
            lambda: self.service.transfer_calibration_correction(
                target,
                self.braggvectors_provider(),
                correction,
            ),
            f"Transfer {label} correction to {target_label}",
            WorkflowStep.CALIBRATION_APPLY,
        )

    def refresh_transfer_targets(self) -> None:
        current_target = self.transfer_target.currentData()
        current_label = self.transfer_target.currentText()
        self.transfer_target.blockSignals(True)
        self.transfer_target.clear()
        if self.transfer_targets_provider is not None:
            for label, braggvectors in self.transfer_targets_provider():
                self.transfer_target.addItem(label, braggvectors)
        if current_target is not None:
            for index in range(self.transfer_target.count()):
                if self.transfer_target.itemData(index) is current_target:
                    self.transfer_target.setCurrentIndex(index)
                    break
        elif current_label:
            index = self.transfer_target.findText(current_label)
            if index >= 0:
                self.transfer_target.setCurrentIndex(index)
        self.transfer_target.blockSignals(False)

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
        self.current_process_name = process_name
        self.current_process_step = process_step
        # operation is a no-arg callable; wrap to accept the progress callback.
        started = self._start_background(
            process_name,
            lambda _cb: operation(),
            parameters={
                "analysis_target": self.analysis_target.currentText(),
                "ellipse_inner": self.ellipse_inner.value(),
                "ellipse_outer": self.ellipse_outer.value(),
                "sampling": self.sampling_spin.value(),
                "q_pixel_size": self.pixel_spin.value(),
                "qr_rotation": self.rotation_spin.value(),
                "transfer_correction": self.transfer_correction.currentText(),
                "transfer_target": self.transfer_target.currentText(),
            },
        )
        if not started:
            for button in self.buttons:
                button.setEnabled(True)
            self.apply_ellipse_button.setEnabled(self.service.pending_ellipse is not None)

    def _handle_result(self, result: CalibrationActionResult) -> None:
        quality = ", ".join(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}" for key, value in result.quality.items())
        self.status_label.setText(
            f"{result.message} ({result.elapsed_seconds:.2f} s)"
            + (f" Quality: {quality}" if quality else "")
        )
        self.log_panel.log(result.message)
        self.log_panel.process_finished(self.current_process_name, result.message)
        if result.images:
            self.viewers.clear()
            self.figure_results.clear()
            self._set_comparison_tab(result)
            for name, image in result.images.items():
                self._set_viewer_tab(
                    name,
                    image,
                    make_current=False,
                    overlay=result.overlays.get(name),
                    vectors=result.vectors.get(name),
                    image_kind=result.image_kinds.get(name),
                )
                if self.result_registry is not None:
                    self.result_registry.register(
                        f"{self.current_process_name} - {name}",
                        "Calibration",
                        image,
                        ("npy", "png", "tiff"),
                        {
                            "process": self.current_process_name,
                            "message": result.message,
                            **self.params_snapshot(),
                        },
                    )
        self._show_measurements(result)
        self.refresh_status(show_histogram=False)
        self.workflow_state.mark_completed(self.current_process_step)

    def _handle_error(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Calibration failed: {message}")
        self.log_panel.process_failed(self.current_process_name, message)
        QMessageBox.warning(self, "Calibration", message)

    def _clear_worker_refs(self) -> None:
        super()._clear_worker_refs()
        for button in self.buttons:
            button.setEnabled(True)
        self.apply_ellipse_button.setEnabled(self.service.pending_ellipse is not None)

    def _float_input(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        unit: str = "",
    ) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(
        self,
        minimum: int,
        maximum: int,
        value: int,
        unit: str = "",
    ) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit, integer=True)

    def _watch_parameters(self) -> None:
        self.workflow_state.watch(
            self.analysis_target, WorkflowStep.CALIBRATION_APPLY, "currentTextChanged"
        )
        for spin in [
            self.ellipse_center_x,
            self.ellipse_center_y,
            self.ellipse_inner,
            self.ellipse_outer,
            self.sampling_spin,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.CALIBRATION_ELLIPSE, "valueChanged")
        self.workflow_state.watch(
            self.pixel_spin, WorkflowStep.CALIBRATION_PIXEL, "valueChanged"
        )
        self.workflow_state.watch(
            self.rotation_spin, WorkflowStep.CALIBRATION_ROTATION, "valueChanged"
        )
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

    def _style_status_labels(self, status, applied: set[str]) -> None:
        for key, label, value, calstate_name in [
            ("origin", self.origin_label, status.origin, "center"),
            ("ellipse", self.ellipse_label, status.ellipse, "ellipse"),
            ("pixel", self.pixel_label, status.pixel, "pixel"),
            ("rotate", self.rotate_label, status.rotate, "rotate"),
        ]:
            if calstate_name in applied:
                color = Theme.READY
            elif value == "missing":
                color = Theme.FAILED if key in self._required_corrections() else Theme.TEXT_DISABLED
            else:
                color = Theme.STALE
            label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _refresh_decision_panel(self) -> None:
        required = self._required_corrections()
        recommended = self._recommended_corrections()
        for key, label in self.decision_labels.items():
            if key in required:
                text = "Required"
                color = Theme.FAILED
            elif key in recommended:
                text = "Recommended"
                color = Theme.STALE
            else:
                text = "Optional"
                color = Theme.TEXT_DISABLED
            label.setText(text)
            label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _required_corrections(self) -> set[str]:
        target = self.analysis_target.currentText()
        if target == "Strain":
            return set()
        if target == "ACOM":
            return {"origin", "pixel"}
        if target == "DPC":
            return {"origin", "rotate"}
        return {"origin"}

    def _recommended_corrections(self) -> set[str]:
        target = self.analysis_target.currentText()
        if target == "Strain":
            return {"origin", "ellipse", "pixel", "rotate"}
        if target == "ACOM":
            return {"ellipse", "rotate"}
        if target == "DPC":
            return {"pixel"}
        if target == "Preview":
            return {"pixel"}
        return set()

    @staticmethod
    def _status_text(value: str, applied: bool) -> str:
        state = "applied" if applied else "not applied"
        return f"{value} [{state}]"

    def _set_viewer_tab(
        self,
        name: str,
        image,
        make_current: bool = False,
        overlay: dict[str, float | str] | None = None,
        vectors: np.ndarray | None = None,
        image_kind: str | None = None,
    ) -> None:
        provider = None
        signed_origin_map = any(
            token in name.lower()
            for token in [
                "qx measured",
                "qx fitted",
                "qx residual",
                "qy measured",
                "qy fitted",
                "qy residual",
            ]
        )
        if "bragg vector" in name.lower() or "braggvectors" in name.lower():
            mode = "cal" if "calibrated" in name.lower() or "corrected" in name.lower() else "raw"
            provider = lambda sampling, mode=mode: np.asarray(
                    self.braggvectors_provider().histogram(mode=mode, sampling=sampling).data
                )
        self.figure_results[name] = FigureResult(
            name,
            image,
            overlay=overlay,
            vectors=vectors,
            image_kind=image_kind or "intensity",
            bragg_sampling_provider=provider,
            key=name,
            colormap="RdBu_r" if signed_origin_map else None,
            scaling="linear" if signed_origin_map else None,
        )
        self.viewers.update_result(name, self.figure_results[name])
        for panel in self.viewers.panels:
            self._connect_annulus_signal(panel.viewer)

    def _set_comparison_tab(self, result: CalibrationActionResult) -> None:
        pairs = self._comparison_images(result.images)
        if len(pairs) < 2:
            return
        for name, image in pairs[:2]:
            self.figure_results[name] = FigureResult(
                name, image, overlay=result.overlays.get(name), key=name
            )
        self.viewers.append_results([self.figure_results[name] for name, _image in pairs[:2]])

    def _comparison_images(self, images: dict[str, object]) -> list[tuple[str, object]]:
        if len(images) < 2:
            return []
        names = list(images.keys())
        bright_field = [name for name in names if "bright-field" in name.lower()]
        if len(bright_field) >= 2:
            return [(name, images[name]) for name in bright_field[:2]]
        raw = [name for name in names if "raw" in name.lower()]
        corrected = [
            name
            for name in names
            if any(token in name.lower() for token in ["calibrated", "corrected", "cal"])
            and name not in raw
        ]
        selected: list[str] = []
        if raw:
            selected.append(raw[0])
        if corrected:
            selected.append(corrected[0])
        for name in names:
            if len(selected) >= 2:
                break
            if name not in selected:
                selected.append(name)
        return [(name, images[name]) for name in selected[:2]]

    def _apply_overlay(self, viewer: ImageViewer, overlay: dict[str, float | str] | None) -> None:
        if overlay is None:
            return
        kind = overlay.get("kind")
        if kind == "circle":
            viewer.set_circle_overlay(
                float(overlay.get("x", 0.0)),
                float(overlay.get("y", 0.0)),
                float(overlay.get("r", 0.0)),
                color="r",
            )
        elif kind == "ring":
            viewer.clear_overlays()
            viewer.add_ring_overlay(
                float(overlay.get("x", 0.0)),
                float(overlay.get("y", 0.0)),
                float(overlay.get("inner_radius", 0.0)),
                float(overlay.get("outer_radius", 0.0)),
                color="r",
            )
            if "a" in overlay and "b" in overlay:
                viewer.add_ellipse_overlay(
                    float(overlay.get("x", 0.0)),
                    float(overlay.get("y", 0.0)),
                    float(overlay.get("a", 0.0)),
                    float(overlay.get("b", 0.0)),
                    float(overlay.get("theta", 0.0)),
                    color="c",
                )
        elif kind == "ellipse":
            viewer.set_ellipse_overlay(
                float(overlay.get("x", 0.0)),
                float(overlay.get("y", 0.0)),
                float(overlay.get("a", 0.0)),
                float(overlay.get("b", 0.0)),
                float(overlay.get("theta", 0.0)),
                color="r",
            )

    def _show_measurements(self, result: CalibrationActionResult) -> None:
        if not result.measurements:
            return
        if self.current_process_step == WorkflowStep.CALIBRATION_ORIGIN:
            self.origin_measurement_label.setText(
                "x={x:.4g}, y={y:.4g}".format(**result.measurements)
            )
        elif self.current_process_step == WorkflowStep.CALIBRATION_ELLIPSE:
            self.ellipse_measurement_label.setText(
                "a={a:.4g}, b={b:.4g}, ellipticity={ellipticity:.4g}".format(
                    **result.measurements
                )
            )

    def start_ellipse_circle_draw(self) -> None:
        self.show_braggvectors_histogram(make_current=True)
        viewer = self._current_image_viewer()
        if viewer is None:
            QMessageBox.information(self, "Ellipse Calibration", "Run full BraggVectors first.")
            return
        viewer.set_interactive_annulus(
            self.ellipse_center_x.value(),
            self.ellipse_center_y.value(),
            self.ellipse_inner.value(),
            self.ellipse_outer.value(),
        )
        self.status_label.setText("Drag the annulus or resize either cyan boundary.")

    def _handle_ellipse_annulus_changed(
        self,
        x: float,
        y: float,
        inner: float,
        outer: float,
    ) -> None:
        for spin, value in [
            (self.ellipse_center_x, x),
            (self.ellipse_center_y, y),
            (self.ellipse_inner, inner),
            (self.ellipse_outer, outer),
        ]:
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self.workflow_state.parameters_updated(WorkflowStep.CALIBRATION_ELLIPSE)
        self.status_label.setText(
            f"Ellipse fit annulus: center=({x:.3g}, {y:.3g}), radii=({inner:.3g}, {outer:.3g})"
        )

    def _connect_annulus_signal(self, viewer: ImageViewer) -> None:
        if bool(viewer.property("calibrationAnnulusConnected")):
            return
        viewer.annulus_changed.connect(self._handle_ellipse_annulus_changed)
        viewer.setProperty("calibrationAnnulusConnected", True)

    def _current_image_viewer(self) -> ImageViewer | None:
        return self.viewers.panels[0].viewer if self.viewers.panels else None

    def _set_default_ellipse_center(self, image) -> None:
        if self.ellipse_center_x.value() or self.ellipse_center_y.value():
            return
        try:
            shape = image.shape
        except Exception:
            return
        self.ellipse_center_x.setValue(max((shape[0] - 1) / 2, 0))
        self.ellipse_center_y.setValue(max((shape[1] - 1) / 2, 0))

    def params_snapshot(self) -> dict[str, object]:
        return {
            "analysis_target": self.analysis_target.currentText(),
            "origin_use_guess": self.origin_use_guess.isChecked(),
            "origin_center_x": self.origin_center_x.value(),
            "origin_center_y": self.origin_center_y.value(),
            "origin_score": self.origin_score.currentText(),
            "origin_find_center": self.origin_find_center.currentText(),
            "origin_fit_function": self.origin_fit_function.currentText(),
            "origin_robust": self.origin_robust.isChecked(),
            "origin_robust_steps": self.origin_robust_steps.value(),
            "origin_robust_threshold": self.origin_robust_threshold.value(),
            "ellipse_center_x": self.ellipse_center_x.value(),
            "ellipse_center_y": self.ellipse_center_y.value(),
            "ellipse_inner": self.ellipse_inner.value(),
            "ellipse_outer": self.ellipse_outer.value(),
            "sampling": self.sampling_spin.value(),
            "q_pixel_size": self.pixel_spin.value(),
            "qr_rotation": self.rotation_spin.value(),
            "rotation_real_direction": self.rotation_real_direction.value(),
            "rotation_real_x": self.rotation_real_x.value(),
            "rotation_real_y": self.rotation_real_y.value(),
            "rotation_q_x": self.rotation_q_x.value(),
            "rotation_q_y": self.rotation_q_y.value(),
            "rotation_real_length": self.rotation_real_length.value(),
            "rotation_q_length": self.rotation_q_length.value(),
            "crystal_cif_path": self.crystal_cif_path,
            "crystal_lattice_parameter": self.crystal_lattice.value(),
            "crystal_atomic_number": int(self.crystal_atomic_number.value()),
            "crystal_k_max": self.crystal_k_max.value(),
            "transfer_correction": self.transfer_correction.currentData(),
            "transfer_target": self.transfer_target.currentText(),
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        if "analysis_target" in params:
            self.analysis_target.setCurrentText(str(params["analysis_target"]))
        for key, spin in [
            ("ellipse_inner", self.ellipse_inner),
            ("ellipse_outer", self.ellipse_outer),
            ("ellipse_center_x", self.ellipse_center_x),
            ("ellipse_center_y", self.ellipse_center_y),
            ("q_pixel_size", self.pixel_spin),
            ("qr_rotation", self.rotation_spin),
            ("origin_center_x", self.origin_center_x),
            ("origin_center_y", self.origin_center_y),
            ("origin_robust_steps", self.origin_robust_steps),
            ("origin_robust_threshold", self.origin_robust_threshold),
            ("rotation_real_direction", self.rotation_real_direction),
            ("rotation_real_x", self.rotation_real_x),
            ("rotation_real_y", self.rotation_real_y),
            ("rotation_q_x", self.rotation_q_x),
            ("rotation_q_y", self.rotation_q_y),
            ("rotation_real_length", self.rotation_real_length),
            ("rotation_q_length", self.rotation_q_length),
            ("crystal_lattice_parameter", self.crystal_lattice),
            ("crystal_atomic_number", self.crystal_atomic_number),
            ("crystal_k_max", self.crystal_k_max),
        ]:
            if key in params:
                spin.setValue(float(params[key]))
        if "origin_use_guess" in params:
            self.origin_use_guess.setChecked(bool(params["origin_use_guess"]))
        if "origin_robust" in params:
            self.origin_robust.setChecked(bool(params["origin_robust"]))
        for key, combo in [
            ("origin_score", self.origin_score),
            ("origin_find_center", self.origin_find_center),
            ("origin_fit_function", self.origin_fit_function),
        ]:
            if key in params:
                combo.setCurrentText(str(params[key]))
        if "sampling" in params:
            self.sampling_spin.setValue(int(params["sampling"]))
        self.crystal_cif_path = str(params.get("crystal_cif_path", self.crystal_cif_path))
        self.crystal_path_label.setText(self.crystal_cif_path or "Manual FCC crystal")
        if "transfer_correction" in params:
            index = self.transfer_correction.findData(str(params["transfer_correction"]))
            if index >= 0:
                self.transfer_correction.setCurrentIndex(index)
        if "transfer_target" in params:
            self.refresh_transfer_targets()
            index = self.transfer_target.findText(str(params["transfer_target"]))
            if index >= 0:
                self.transfer_target.setCurrentIndex(index)
