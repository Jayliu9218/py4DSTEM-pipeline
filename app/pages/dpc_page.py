from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.phase_contrast_service import (
    DPCAcceptanceState,
    DPCPreprocessParams,
    DPCReconstructionParams,
    DPCSegmentedParams,
    DPCStageResult,
    PhaseContrastResult,
    PhaseContrastService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class DPCPage(QWidget, WorkerRunner):
    dpc_result_ready = Signal(object)

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
        service: PhaseContrastService | None = None,
        stage_mode: str = "all",
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = service or PhaseContrastService()
        self.stage_mode = stage_mode
        self._init_worker_runner()
        self.result: PhaseContrastResult | None = None
        self.segmented_result: DPCStageResult | None = None

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.rotation_label = QLabel("Preprocessing not run")
        self.rotation_label.setWordWrap(True)

        self.preset_button = QPushButton("Apply 02_DPC STO Preset")
        self.preset_button.clicked.connect(self.apply_sto_preset)
        self.energy = self._float_input(10, 1_000_000, 200_000, decimals=0, unit="eV")

        self.segment_sampling_x = self._float_input(1e-6, 1000, 0.246570625, decimals=9, unit="A")
        self.segment_sampling_y = self._float_input(1e-6, 1000, 0.246570625, decimals=9, unit="A")
        self.segment_rotation = self._float_input(-360, 360, 60, unit="deg")
        self.segment_inner = self._float_input(0, 1000, 10, unit="mrad")
        self.segment_outer = self._float_input(0, 1000, 25, unit="mrad")
        self.segment_center_x = self._float_input(-1, 100000, -1, unit="px")
        self.segment_center_y = self._float_input(-1, 100000, -1, unit="px")
        self.segment_button = QPushButton("Run Four-Segment Demonstration")
        self.segment_button.clicked.connect(self.run_segmented)
        self.segment_view = QComboBox()
        self.segment_view.addItems(["Detector masks", "Segment intensities", "CoM results"])
        self.segment_view.currentTextChanged.connect(self._refresh_segmented_display)

        self.use_dp_mask = QCheckBox("Use annular diffraction mask")
        self.mask_inner = self._float_input(0, 1000, 0, unit="mrad")
        self.mask_outer = self._float_input(0, 1000, 25, unit="mrad")
        self.padding_factor = self._float_input(1, 10, 2, decimals=2)
        self.rotation_start = self._float_input(-360, 360, -90, unit="deg")
        self.rotation_end = self._float_input(-360, 360, 90, unit="deg")
        self.rotation_step = self._float_input(0.01, 180, 1, unit="deg")
        self.rotation_metric = QComboBox()
        self.rotation_metric.addItems(["Minimize curl", "Maximize divergence"])
        self.fit_function = QComboBox()
        self.fit_function.addItems(["plane", "parabola", "bezier_two"])
        self.rotation_mode = QComboBox()
        self.rotation_mode.addItems(["Auto", "Force"])
        self.force_rotation = self._float_input(-360, 360, 0, unit="deg")
        self.transpose_mode = QComboBox()
        self.transpose_mode.addItems(["Auto", "Force off", "Force on"])
        self.manual_shift = QCheckBox("Use manual CoM shifts")
        self.shift_x = self._float_input(-100000, 100000, 0, unit="px")
        self.shift_y = self._float_input(-100000, 100000, 0, unit="px")
        self.vectorized = QCheckBox("Vectorized CoM calculation")
        self.preprocess_button = QPushButton("Calculate Pixelated CoM")
        self.preprocess_button.clicked.connect(self.run_preprocess)
        self.accept_button = QPushButton("Accept CoM Preprocessing")
        self.accept_button.clicked.connect(self.accept_preprocessing)
        self.accept_button.setEnabled(False)
        self.preprocess_view = QComboBox()
        self.preprocess_view.addItems(["CoM diagnostics", "Rotation review"])
        self.preprocess_view.currentTextChanged.connect(self._refresh_preprocess_display)

        self.reset_reconstruction = QCheckBox("Reset reconstruction")
        self.reset_reconstruction.setChecked(True)
        self.max_iter = self._int_input(1, 10000, 64, unit="iter")
        self.use_step_size = QCheckBox("Set initial step size")
        self.step_size = self._float_input(1e-12, 1000, 0.5, decimals=6)
        self.stopping = self._float_input(1e-12, 1, 1e-6, decimals=9)
        self.backtrack = QCheckBox("Backtrack")
        self.backtrack.setChecked(True)
        self.gaussian_filter = QCheckBox("Gaussian filter")
        self.gaussian_filter.setChecked(True)
        self.use_gaussian_sigma = QCheckBox("Set Gaussian sigma")
        self.gaussian_sigma = self._float_input(0, 1000, 0, unit="A")
        self.butterworth_filter = QCheckBox("Butterworth filter")
        self.butterworth_filter.setChecked(True)
        self.use_q_lowpass = QCheckBox("Set Q lowpass")
        self.q_lowpass = self._float_input(0, 1000, 0, decimals=4, unit="A^-1")
        self.use_q_highpass = QCheckBox("Set Q highpass")
        self.q_highpass = self._float_input(0, 1000, 0, decimals=4, unit="A^-1")
        self.butterworth_order = self._float_input(0.1, 100, 2, decimals=2)
        self.store_iterations = QCheckBox("Store iterations")
        self.reconstruct_button = QPushButton("Reconstruct Integrated CoM")
        self.reconstruct_button.clicked.connect(self.run_reconstruction)
        self.reconstruct_button.setEnabled(False)

        self.workspace = AdaptiveImageWorkspace()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()
        self._watch_parameters()
        self._configure_stage_mode()

    def _build_layout(self) -> None:
        self.input_group = QGroupBox("Input / Preset")
        input_form = QFormLayout(self.input_group)
        input_form.addRow(self.preset_button)
        input_form.addRow("Energy", self.energy)

        self.segmented_group = QGroupBox("Four-Segment DPC Demonstration")
        segmented_form = QFormLayout(self.segmented_group)
        segmented_form.addRow("Sampling x", self.segment_sampling_x)
        segmented_form.addRow("Sampling y", self.segment_sampling_y)
        segmented_form.addRow("Rotation offset", self.segment_rotation)
        segmented_form.addRow("Inner radius", self.segment_inner)
        segmented_form.addRow("Outer radius", self.segment_outer)
        segmented_form.addRow("Center x (-1 auto)", self.segment_center_x)
        segmented_form.addRow("Center y (-1 auto)", self.segment_center_y)
        segmented_form.addRow(self.segment_button)
        segmented_form.addRow("Review", self.segment_view)

        self.preprocess_group = QGroupBox("Pixelated CoM Preprocessing")
        preprocess_form = QFormLayout(self.preprocess_group)
        preprocess_form.addRow(self.use_dp_mask)
        preprocess_form.addRow("Mask inner", self.mask_inner)
        preprocess_form.addRow("Mask outer", self.mask_outer)
        preprocess_form.addRow("Padding factor", self.padding_factor)
        preprocess_form.addRow("Rotation start", self.rotation_start)
        preprocess_form.addRow("Rotation end", self.rotation_end)
        preprocess_form.addRow("Rotation step", self.rotation_step)
        preprocess_form.addRow("Rotation metric", self.rotation_metric)
        preprocess_form.addRow("CoM fit", self.fit_function)
        preprocess_form.addRow("Rotation mode", self.rotation_mode)
        preprocess_form.addRow("Forced rotation", self.force_rotation)
        preprocess_form.addRow("Transpose", self.transpose_mode)
        preprocess_form.addRow(self.manual_shift)
        preprocess_form.addRow("Shift x", self.shift_x)
        preprocess_form.addRow("Shift y", self.shift_y)
        preprocess_form.addRow(self.vectorized)
        preprocess_form.addRow(self.preprocess_button)

        self.reconstruct_group = QGroupBox("Integrated CoM Reconstruction")
        reconstruct_form = QFormLayout(self.reconstruct_group)
        reconstruct_form.addRow(self.reset_reconstruction)
        reconstruct_form.addRow("Maximum iterations", self.max_iter)
        reconstruct_form.addRow(self.use_step_size)
        reconstruct_form.addRow("Initial step size", self.step_size)
        reconstruct_form.addRow("Stopping criterion", self.stopping)
        reconstruct_form.addRow(self.backtrack)
        reconstruct_form.addRow(self.gaussian_filter)
        reconstruct_form.addRow(self.use_gaussian_sigma)
        reconstruct_form.addRow("Gaussian sigma", self.gaussian_sigma)
        reconstruct_form.addRow(self.butterworth_filter)
        reconstruct_form.addRow(self.use_q_lowpass)
        reconstruct_form.addRow("Q lowpass", self.q_lowpass)
        reconstruct_form.addRow(self.use_q_highpass)
        reconstruct_form.addRow("Q highpass", self.q_highpass)
        reconstruct_form.addRow("Butterworth order", self.butterworth_order)
        reconstruct_form.addRow(self.store_iterations)
        reconstruct_form.addRow(self.reconstruct_button)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        for group in (
            self.input_group,
            self.segmented_group,
            self.preprocess_group,
            self.reconstruct_group,
        ):
            layout.addWidget(group)
        layout.addWidget(self.rotation_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.controls_panel = controls

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.workspace, 1)

    def _configure_stage_mode(self) -> None:
        visible = {
            "all": {"input", "segmented", "preprocess", "reconstruct"},
            "segmented": {"input", "segmented"},
            "preprocess": {"input", "preprocess"},
            "review": set(),
            "reconstruct": {"reconstruct"},
        }.get(self.stage_mode, set())
        self.input_group.setVisible("input" in visible)
        self.segmented_group.setVisible("segmented" in visible)
        self.preprocess_group.setVisible("preprocess" in visible)
        self.reconstruct_group.setVisible("reconstruct" in visible)
        self.accept_button.setVisible(False)
        self.preprocess_view.setVisible(False)
        if self.stage_mode in {"preprocess", "review", "all"}:
            self.accept_button.setText("Accept CoM Preprocessing")
            self.accept_button.setVisible(True)
            self.preprocess_view.setVisible(True)
            review_controls = QGroupBox("CoM Review & Acceptance")
            form = QFormLayout(review_controls)
            form.addRow("Review", self.preprocess_view)
            form.addRow(self.accept_button)

            self.controls_panel.layout().insertWidget(3, review_controls)

        self.refresh_stage()

    def refresh_stage(self) -> None:
        if self.stage_mode in {"preprocess", "review"}:
            self.result = self.service.dpc_preprocess_result
            self.accept_button.setEnabled(self.result is not None)
            if self.result is not None:
                self._refresh_preprocess_display()
                self._update_rotation_label(self.result)
        elif self.stage_mode == "reconstruct":
            self.reconstruct_button.setEnabled(self.service.dpc_acceptance.preprocessing)

    def apply_sto_preset(self) -> None:
        preset = self.service.dpc_sto_preset()
        self.energy.setValue(preset.energy)
        self.segment_sampling_x.setValue(preset.sampling_x)
        self.segment_sampling_y.setValue(preset.sampling_y)
        self.segment_rotation.setValue(preset.rotation_offset_degrees)
        self.segment_inner.setValue(preset.inner_radius_mrad)
        self.segment_outer.setValue(preset.outer_radius_mrad)
        source = self.source_provider()
        shape = getattr(source, "shape", None)
        if shape is not None and len(shape) >= 4:
            self.segment_center_x.setValue(float(shape[-2]) / 2)
            self.segment_center_y.setValue(float(shape[-1]) / 2)
        sampling = self._source_sampling(source)
        if sampling is not None:
            self.segment_sampling_x.setValue(sampling[0])
            self.segment_sampling_y.setValue(sampling[1])
        self.status_label.setText("02_DPC STO preset applied.")

    def run_segmented(self) -> None:
        source = self._source_or_warn()
        if source is None:
            return
        self._start_operation(
            "Segmented DPC",
            lambda: self.service.generate_segmented_dpc(source, self._segmented_params()),
        )

    def run_preprocess(self) -> None:
        source = self._source_or_warn()
        if source is None:
            return
        if not self.workflow_state.is_completed(WorkflowStep.BF_DF_PREVIEW):
            self.log_panel.log("WARN  BF/DF Preview is recommended before DPC mask review.")
        self._start_operation(
            "DPC CoM preprocessing",
            lambda: self.service.preprocess_dpc(source, self._preprocess_params()),
        )

    def accept_preprocessing(self) -> None:
        try:
            state = self.service.accept_dpc_preprocessing()
        except Exception as exc:
            self._handle_error(str(exc))
            return
        self.reconstruct_button.setEnabled(state.preprocessing)
        self.status_label.setText("Pixelated CoM preprocessing accepted.")
        self.log_panel.log("DPC pixelated CoM preprocessing accepted.")
        self.workflow_state.mark_completed(WorkflowStep.DPC_REVIEW)

    def run_reconstruction(self) -> None:
        self._start_operation(
            "DPC reconstruction",
            lambda: self.service.reconstruct_dpc(self._reconstruction_params()),
        )

    def _start_operation(self, name: str, operation) -> None:
        for button in (self.segment_button, self.preprocess_button, self.reconstruct_button):
            button.setEnabled(False)
        # operation is a no-arg lambda; wrap to accept the progress callback.
        self._start_background(name, lambda _cb: operation(), parameters=self.params_snapshot())

    def _handle_result(self, result: object) -> None:
        if isinstance(result, DPCStageResult):
            self.segmented_result = result
            self.segment_view.setCurrentText("Detector masks")
            self._refresh_segmented_display()
            self._register_stage(result)
            self.workflow_state.mark_completed(WorkflowStep.DPC_SEGMENTED)
        elif isinstance(result, PhaseContrastResult):
            self.result = result
            is_reconstruction = self.pending_operation == "DPC reconstruction"
            if is_reconstruction or self.stage_mode != "review":
                figures = self._result_figures(result, final=is_reconstruction)
                self.workspace.set_results(figures)
            self._register_result(result)
            self._update_rotation_label(result)
            if is_reconstruction:
                self.dpc_result_ready.emit(result)
                self.workflow_state.mark_completed(WorkflowStep.DPC)
            else:
                self.service.dpc_preprocess_result = result
                self.workflow_state.mark_completed(WorkflowStep.DPC_PREPROCESS)
        elapsed = getattr(result, "elapsed_seconds", 0.0)
        if self.stage_mode == "preprocess" and isinstance(result, PhaseContrastResult):
            self.status_label.setText(
                f"{self.pending_operation} complete in {elapsed:.2f} s. "
                "Review the diagnostics below and explicitly accept preprocessing."
            )
        else:
            self.status_label.setText(f"{self.pending_operation} complete in {elapsed:.2f} s")
        self.log_panel.process_finished(self.pending_operation, f"elapsed={elapsed:.2f} s")

    def _update_rotation_label(self, result: PhaseContrastResult) -> None:
        if result.rotation_degrees is not None:
            self.rotation_label.setText(
                f"Best rotation: {result.rotation_degrees:.2f} deg; "
                f"transpose={result.transpose if result.transpose is not None else 'unknown'}"
            )

    def _refresh_segmented_display(self, *_args) -> None:
        result = self.segmented_result
        if result is None:
            return
        view = self.segment_view.currentText()
        if view == "Detector masks":
            mean_dp = result.images["Mean diffraction pattern"]
            figures = [
                FigureResult(
                    f"Segment {index + 1} detector position",
                    mean_dp,
                    scaling="log",
                    colormap="gray",
                    mask=mask,
                )
                for index, mask in enumerate(result.masks)
            ]
        elif view == "Segment intensities":
            figures = [
                FigureResult(name, result.images[name], scaling="linear")
                for name in result.images
                if name.endswith(" intensity")
            ]
        else:
            names = ["Segmented CoM X", "Segmented CoM Y", "Weighted CoM X", "Weighted CoM Y"]
            figures = [
                FigureResult(name, result.images[name], scaling="linear", colormap="RdBu_r")
                for name in names
            ]
            figures.extend(
                FigureResult(name, self._complex_rgb(image), image_kind="rgb")
                for name, image in result.complex_images.items()
            )
        self.workspace.set_results(figures[:6])

    def _refresh_preprocess_display(self, *_args) -> None:
        if self.result is None:
            return
        self.workspace.set_results(self._result_figures(self.result, final=False))

    def _result_figures(self, result: PhaseContrastResult, final: bool) -> list[FigureResult]:
        if final:
            names = ["Potential"] + [
                name for name in result.images if name.startswith("Iteration ")
            ]
        else:
            if self.preprocess_view.currentText() == "Rotation review":
                figures = []
                if result.rotation_angles_degrees is not None and result.rotation_metric is not None:
                    figures.append(
                        FigureResult(
                            "Rotation curl / divergence search",
                            self._curve_rgb(
                                result.rotation_angles_degrees,
                                result.rotation_metric,
                                result.rotation_metric_transpose,
                            ),
                            image_kind="rgb",
                            diagnostic=(
                                f"Best rotation {result.rotation_degrees:.2f} deg; "
                                f"transpose={result.transpose}"
                                if result.rotation_degrees is not None
                                else ""
                            ),
                        )
                    )
                for name in ("CoM X", "CoM Y"):
                    if name in result.images:
                        figures.append(
                            FigureResult(name, result.images[name], scaling="linear", colormap="RdBu_r")
                        )
                if result.complex_com is not None:
                    figures.append(
                        FigureResult(
                            "Corrected complex CoM",
                            self._complex_rgb(result.complex_com),
                            image_kind="rgb",
                        )
                    )
                return figures
            names = [
                "Measured CoM X", "Measured CoM Y", "Fitted CoM X",
                "Fitted CoM Y", "Normalized CoM X", "Normalized CoM Y",
            ]
        figures = [
            FigureResult(name, result.images[name], scaling="linear", colormap="RdBu_r")
            for name in names
            if name in result.images
        ]
        if final and result.error_iterations is not None and result.error_iterations.size:
            iterations = np.arange(result.error_iterations.size)
            figures.insert(
                1,
                FigureResult(
                    "Convergence error",
                    self._curve_rgb(iterations, result.error_iterations),
                    image_kind="rgb",
                ),
            )
        if not final and result.complex_com is not None:
            figures.append(
                FigureResult(
                    "Corrected complex CoM",
                    self._complex_rgb(result.complex_com),
                    image_kind="rgb",
                )
            )
        return figures[:6]

    @staticmethod
    def _curve_rgb(
        x: np.ndarray,
        primary: np.ndarray,
        transpose: np.ndarray | None = None,
    ) -> np.ndarray:
        figure = Figure(figsize=(6, 4), dpi=100, tight_layout=True)
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(x, primary, label="normal")
        if transpose is not None:
            axis.plot(x, transpose, label="transposed")
        axis.set_xlabel("Rotation (deg)")
        axis.set_ylabel("Metric")
        axis.legend()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    @staticmethod
    def _complex_rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        amplitude = np.abs(array)
        high = float(np.nanpercentile(amplitude, 99)) if amplitude.size else 1.0
        value = np.clip(amplitude / max(high, 1e-12), 0, 1)
        hue = (np.angle(array) + np.pi) / (2 * np.pi)
        hsv = np.stack([hue, np.ones_like(value), value], axis=-1)
        return np.asarray(np.round(hsv_to_rgb(hsv) * 255), dtype=np.uint8)

    def _register_stage(self, result: DPCStageResult) -> None:
        if self.result_registry is None:
            return
        for name, image in result.images.items():
            self.result_registry.register(
                f"dpc_{name.lower().replace(' ', '_')}",
                "Phase Retrieval",
                image,
                ("npy", "png", "tiff"),
                result.metadata,
            )
        for name, image in result.complex_images.items():
            self.result_registry.register(
                f"dpc_{name.lower().replace(' ', '_')}",
                "Phase Retrieval",
                image,
                ("npy",),
                result.metadata,
            )

    def _register_result(self, result: PhaseContrastResult) -> None:
        if self.result_registry is None:
            return
        for name, image in result.images.items():
            self.result_registry.register(
                f"dpc_{name.lower().replace(' ', '_')}",
                "Phase Retrieval",
                image,
                ("npy", "png", "tiff"),
                self.params_snapshot(),
            )
        if result.complex_com is not None:
            self.result_registry.register(
                "dpc_complex_com",
                "Phase Retrieval",
                result.complex_com,
                ("npy",),
                self.params_snapshot(),
            )
        for key, value in (
            ("dpc_rotation_angles", result.rotation_angles_degrees),
            ("dpc_rotation_metric", result.rotation_metric),
            ("dpc_rotation_metric_transpose", result.rotation_metric_transpose),
            ("dpc_convergence_error", result.error_iterations),
        ):
            if value is not None:
                self.result_registry.register(
                    key, "Phase Retrieval", value, ("npy",), self.params_snapshot()
                )

    def _segmented_params(self) -> DPCSegmentedParams:
        optional_center = lambda value: None if value < 0 else value
        return DPCSegmentedParams(
            energy=self.energy.value(),
            sampling_x=self.segment_sampling_x.value(),
            sampling_y=self.segment_sampling_y.value(),
            rotation_offset_degrees=self.segment_rotation.value(),
            inner_radius_mrad=self.segment_inner.value(),
            outer_radius_mrad=self.segment_outer.value(),
            center_x=optional_center(self.segment_center_x.value()),
            center_y=optional_center(self.segment_center_y.value()),
        )

    def _preprocess_params(self) -> DPCPreprocessParams:
        transpose = {"Auto": None, "Force off": False, "Force on": True}[
            self.transpose_mode.currentText()
        ]
        return DPCPreprocessParams(
            energy=self.energy.value(),
            padding_factor=self.padding_factor.value(),
            rotation_start_degrees=self.rotation_start.value(),
            rotation_end_degrees=self.rotation_end.value(),
            rotation_step_degrees=self.rotation_step.value(),
            maximize_divergence=self.rotation_metric.currentText() == "Maximize divergence",
            fit_function=self.fit_function.currentText(),
            force_com_rotation=(
                self.force_rotation.value() if self.rotation_mode.currentText() == "Force" else None
            ),
            force_com_transpose=transpose,
            force_com_shift_x=self.shift_x.value() if self.manual_shift.isChecked() else None,
            force_com_shift_y=self.shift_y.value() if self.manual_shift.isChecked() else None,
            vectorized_com_calculation=self.vectorized.isChecked(),
            use_dp_mask=self.use_dp_mask.isChecked(),
            mask_inner_mrad=self.mask_inner.value(),
            mask_outer_mrad=self.mask_outer.value(),
        )

    def _reconstruction_params(self) -> DPCReconstructionParams:
        return DPCReconstructionParams(
            reset=self.reset_reconstruction.isChecked(),
            max_iter=int(self.max_iter.value()),
            step_size=self.step_size.value() if self.use_step_size.isChecked() else None,
            stopping_criterion=self.stopping.value(),
            backtrack=self.backtrack.isChecked(),
            gaussian_filter=self.gaussian_filter.isChecked(),
            gaussian_filter_sigma=(
                self.gaussian_sigma.value() if self.use_gaussian_sigma.isChecked() else None
            ),
            butterworth_filter=self.butterworth_filter.isChecked(),
            q_lowpass=self.q_lowpass.value() if self.use_q_lowpass.isChecked() else None,
            q_highpass=self.q_highpass.value() if self.use_q_highpass.isChecked() else None,
            butterworth_order=self.butterworth_order.value(),
            store_iterations=self.store_iterations.isChecked(),
        )

    def _source_or_warn(self):
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "DPC / CoM", "Load a 4D DataCube first.")
        return source

    @staticmethod
    def _source_sampling(source: object | None) -> tuple[float, float] | None:
        calibration = getattr(source, "calibration", None)
        if calibration is None:
            return None
        try:
            sampling = calibration.get_R_pixel_size()
            if np.isscalar(sampling):
                value = float(sampling)
                return value, value
            values = tuple(float(item) for item in sampling)
            return values[:2] if len(values) >= 2 else None
        except Exception:
            return None

    def _handle_error(self, error: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"{self.pending_operation or 'DPC'} failed: {error}")
        self.log_panel.process_finished(f"{self.pending_operation or 'DPC'} failed", error)
        self._enable_available_buttons()

    def _handle_progress(self, message: str, fraction: float) -> None:
        self.log_panel.log(message)
        super()._handle_progress(message, fraction)

    def _clear_worker_refs(self) -> None:
        super()._clear_worker_refs()
        self._enable_available_buttons()

    def _enable_available_buttons(self) -> None:
        self.segment_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.service.dpc_acceptance.preprocessing)

    def _watch_parameters(self) -> None:
        step = {
            "segmented": WorkflowStep.DPC_SEGMENTED,
            "preprocess": WorkflowStep.DPC_PREPROCESS,
            "review": WorkflowStep.DPC_REVIEW,
            "reconstruct": WorkflowStep.DPC,
        }.get(self.stage_mode, WorkflowStep.DPC)
        for control in self.findChildren(NumericLineEdit):
            self.workflow_state.watch(control, step, "valueChanged")
        for control in self.findChildren(QComboBox):
            self.workflow_state.watch(control, step, "currentTextChanged")
        for control in self.findChildren(QCheckBox):
            self.workflow_state.watch(control, step, "toggled")
        preprocess_controls = [
            self.energy, self.use_dp_mask, self.mask_inner, self.mask_outer,
            self.padding_factor, self.rotation_start, self.rotation_end,
            self.rotation_step, self.rotation_metric, self.fit_function,
            self.rotation_mode, self.force_rotation, self.transpose_mode,
            self.manual_shift, self.shift_x, self.shift_y, self.vectorized,
        ]
        for control in preprocess_controls:
            signal = (
                control.valueChanged if isinstance(control, NumericLineEdit)
                else control.currentTextChanged if isinstance(control, QComboBox)
                else control.toggled
            )
            signal.connect(self._invalidate_preprocess_acceptance)

    def _invalidate_preprocess_acceptance(self, *_args) -> None:
        if not self.service.dpc_acceptance.preprocessing:
            return
        self.service.dpc_acceptance = DPCAcceptanceState(False)
        self.reconstruct_button.setEnabled(False)
        self.workflow_state.parameters_updated(WorkflowStep.DPC_PREPROCESS)
        self.status_label.setText("CoM preprocessing parameters changed; review and accept again.")

    def _refresh_stale_status(self) -> None:
        step = {
            "segmented": WorkflowStep.DPC_SEGMENTED,
            "preprocess": WorkflowStep.DPC_PREPROCESS,
            "review": WorkflowStep.DPC_REVIEW,
            "reconstruct": WorkflowStep.DPC,
        }.get(self.stage_mode, WorkflowStep.DPC)
        if self.workflow_state.is_stale(step):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(self, minimum, maximum, value, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit, integer=True)

    def params_snapshot(self) -> dict[str, object]:
        segmented = self._segmented_params()
        preprocess = self._preprocess_params()
        reconstruction = self._reconstruction_params()
        return {
            **{f"segment_{key}": value for key, value in segmented.__dict__.items()},
            **{f"preprocess_{key}": value for key, value in preprocess.__dict__.items()},
            **{f"reconstruct_{key}": value for key, value in reconstruction.__dict__.items()},
            "preprocessing_accepted": self.service.dpc_acceptance.preprocessing,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        numeric = {
            "segment_energy": self.energy,
            "dpc_energy": self.energy,
            "segment_sampling_x": self.segment_sampling_x,
            "segment_sampling_y": self.segment_sampling_y,
            "segment_rotation_offset_degrees": self.segment_rotation,
            "segment_inner_radius_mrad": self.segment_inner,
            "segment_outer_radius_mrad": self.segment_outer,
            "segment_center_x": self.segment_center_x,
            "segment_center_y": self.segment_center_y,
            "preprocess_padding_factor": self.padding_factor,
            "dpc_padding": self.padding_factor,
            "preprocess_rotation_start_degrees": self.rotation_start,
            "preprocess_rotation_end_degrees": self.rotation_end,
            "preprocess_rotation_step_degrees": self.rotation_step,
            "preprocess_mask_inner_mrad": self.mask_inner,
            "preprocess_mask_outer_mrad": self.mask_outer,
            "preprocess_force_com_rotation": self.force_rotation,
            "preprocess_force_com_shift_x": self.shift_x,
            "preprocess_force_com_shift_y": self.shift_y,
            "reconstruct_max_iter": self.max_iter,
            "reconstruct_step_size": self.step_size,
            "reconstruct_stopping_criterion": self.stopping,
            "reconstruct_gaussian_filter_sigma": self.gaussian_sigma,
            "reconstruct_q_lowpass": self.q_lowpass,
            "reconstruct_q_highpass": self.q_highpass,
            "reconstruct_butterworth_order": self.butterworth_order,
        }
        for key, control in numeric.items():
            if key in params:
                value = params[key]
                if key == "dpc_padding":
                    value = 2
                    self.log_panel.log("Migrated legacy DPC pixel padding to padding factor 2.")
                if value is not None:
                    control.setValue(float(value))
        combos = {
            "preprocess_maximize_divergence": (
                self.rotation_metric,
                lambda value: "Maximize divergence" if value else "Minimize curl",
            ),
            "preprocess_fit_function": (self.fit_function, str),
            "preprocess_force_com_rotation": (
                self.rotation_mode,
                lambda value: "Force" if value is not None else "Auto",
            ),
            "preprocess_force_com_transpose": (
                self.transpose_mode,
                lambda value: "Auto" if value is None else "Force on" if value else "Force off",
            ),
        }
        for key, (control, formatter) in combos.items():
            if key in params:
                control.setCurrentText(formatter(params[key]))
        checks = {
            "preprocess_use_dp_mask": self.use_dp_mask,
            "preprocess_vectorized_com_calculation": self.vectorized,
            "reconstruct_reset": self.reset_reconstruction,
            "reconstruct_backtrack": self.backtrack,
            "reconstruct_gaussian_filter": self.gaussian_filter,
            "reconstruct_butterworth_filter": self.butterworth_filter,
            "reconstruct_store_iterations": self.store_iterations,
        }
        for key, control in checks.items():
            if key in params:
                control.setChecked(bool(params[key]))
        self.manual_shift.setChecked(
            params.get("preprocess_force_com_shift_x") is not None
            and params.get("preprocess_force_com_shift_y") is not None
        )
        self.use_step_size.setChecked(params.get("reconstruct_step_size") is not None)
        self.use_gaussian_sigma.setChecked(
            params.get("reconstruct_gaussian_filter_sigma") is not None
        )
        self.use_q_lowpass.setChecked(params.get("reconstruct_q_lowpass") is not None)
        self.use_q_highpass.setChecked(params.get("reconstruct_q_highpass") is not None)
        if bool(params.get("preprocessing_accepted", False)) and self.service.dpc is not None:
            self.service.dpc_acceptance = DPCAcceptanceState(True)
            self.reconstruct_button.setEnabled(True)
