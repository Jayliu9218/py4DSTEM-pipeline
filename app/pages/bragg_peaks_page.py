from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import (
    BraggDetectionParams,
    BraggStrainService,
    BraggVectorsResult,
    CBSPreset,
    PeakDetectionResult,
    ProbeKernelResult,
    SelectedPeaksResult,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class BraggPeaksPage(QWidget, WorkerRunner):
    braggvectors_ready = Signal()

    def __init__(
        self,
        datacube_provider: Callable[[], object | None],
        shape_provider: Callable[[], tuple[int, int, int, int] | None],
        virtual_image_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.shape_provider = shape_provider
        self.virtual_image_provider = virtual_image_provider
        self.service = service
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.cuda_enabled = False
        self._init_worker_runner()
        self._pending_result_handler: Callable[[object], None] | None = None
        self._pending_status: str = ""
        self.roi_pick_points: list[tuple[int, int]] = []

        self.rx_spin = self._int_input(0, 100000, 0, unit="px")
        self.ry_spin = self._int_input(0, 100000, 0, unit="px")
        self.min_abs_spin = self._float_input(0, 1e12, 2, unit="int.")
        self.min_rel_spin = self._float_input(0, 1, 0, decimals=4, unit="ratio")
        self.spacing_spin = self._int_input(1, 10000, 18, unit="px")
        self.edge_spin = self._int_input(0, 10000, 2, unit="px")
        self.max_peaks_spin = self._int_input(1, 10000, 100, unit="peaks")
        self.sigma_spin = self._float_input(0, 1000, 0, unit="px")
        self.sigma_dp_spin = self._float_input(0, 1000, 0, unit="px")
        self.corr_power_spin = self._float_input(0, 1, 1, decimals=3)
        self.upsample_spin = self._int_input(1, 1024, 16, unit="x")
        self.radial_background_check = QCheckBox("Enable")
        self.gaussian_fallback_check = QCheckBox("Allow with warning")
        self.selection_mode = QComboBox()
        self.selection_mode.addItems(["Deterministic positions", "Seeded random"])
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("01_CBS Au")
        self.apply_preset_button = QPushButton("Apply Preset")
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["poly", "multicorr", "pixel"])
        self.roi_rx_start = self._int_input(0, 100000, 0, unit="px")
        self.roi_rx_end = self._int_input(0, 100000, 1, unit="px")
        self.roi_ry_start = self._int_input(0, 100000, 0, unit="px")
        self.roi_ry_end = self._int_input(0, 100000, 1, unit="px")

        self.prepare_kernel_button = QPushButton("Prepare Probe Kernel")
        self.pick_roi_button = QPushButton("Draw ROI")
        self.run_current_button = QPushButton("Run Current Pattern")
        self.run_selected_button = QPushButton("Check 6 Selected Positions")
        self.run_full_button = QPushButton("Run Full BraggVectors")
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.count_label = QLabel("Peaks: -")
        self.quality_label = QLabel("Quality: not calculated")
        self.quality_label.setWordWrap(True)
        self.viewer = ImageViewer()
        self.roi_viewer = ImageViewer()
        self.workspace = AdaptiveImageWorkspace()
        self.selected_grid = self.workspace
        self.full_map_viewer = ImageViewer()
        self.full_map_viewer.set_bragg_sampling_provider(self._sampled_bragg_vector_map)
        self.clear_results()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["index", "qx", "qy", "intensity", "distance"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.prepare_kernel_button.clicked.connect(self.prepare_probe_kernel)
        self.pick_roi_button.clicked.connect(self.start_roi_pick)
        self.roi_viewer.image_clicked.connect(self._handle_roi_click)
        self.roi_viewer.roi_changed.connect(self._handle_drawn_roi_changed)
        self.run_current_button.clicked.connect(self.run_current_pattern)
        self.run_selected_button.clicked.connect(self.run_selected_positions)
        self.run_full_button.clicked.connect(self.run_full_braggvectors)
        self.apply_preset_button.clicked.connect(self.apply_au_preset)
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def clear_results(self) -> None:
        self.workspace.set_results([
            FigureResult("Probe ROI", np.zeros((2, 2)), key="probe-roi", viewer=self.roi_viewer),
            FigureResult("Single Position", np.zeros((2, 2)), key="single-position", viewer=self.viewer),
            FigureResult(
                "Full Bragg Vector Map",
                np.zeros((2, 2)),
                key="full-map",
                viewer=self.full_map_viewer,
                bragg_sampling_provider=self._sampled_bragg_vector_map,
            ),
        ])
        self.roi_viewer.clear("Run a virtual image first, then select a probe ROI here.")
        self.viewer.clear("Run current pattern to detect Bragg peaks.")
        self.full_map_viewer.clear("Run full BraggVectors to display the map.")

    def set_cuda_enabled(self, enabled: bool) -> None:
        self.cuda_enabled = enabled

    def set_virtual_image(self, image) -> None:
        if image is None:
            self.roi_viewer.clear("Run a virtual image first, then select a probe ROI here.")
            return
        self.roi_viewer.set_image(image)
        self.workspace.update_result(
            "probe-roi",
            FigureResult("Probe ROI", image, viewer=self.roi_viewer),
        )
        self._update_roi_overlay()
        self.log_panel.log("Virtual image sent to Probe ROI selection.")

    def refresh_from_datacube(self) -> None:
        shape = self.shape_provider()
        if shape is None:
            return
        self.rx_spin.setMaximum(max(shape[0] - 1, 0))
        self.ry_spin.setMaximum(max(shape[1] - 1, 0))
        self.rx_spin.setValue(max((shape[0] - 1) // 2, 0))
        self.ry_spin.setValue(max((shape[1] - 1) // 2, 0))
        for spin, maximum in [
            (self.roi_rx_start, shape[0] - 1),
            (self.roi_rx_end, shape[0]),
            (self.roi_ry_start, shape[1] - 1),
            (self.roi_ry_end, shape[1]),
        ]:
            spin.setRange(0, max(maximum, 0))
        self.roi_rx_start.setValue(0)
        self.roi_ry_start.setValue(0)
        self.roi_rx_end.setValue(max(min(5, shape[0]), 1))
        self.roi_ry_end.setValue(max(min(5, shape[1]), 1))
        self._sync_virtual_image()
        self._update_roi_overlay()
        self.log_panel.log("Bragg Peaks controls updated from current DataCube.")

    def prepare_probe_kernel(self) -> None:
        datacube = self.datacube_provider()
        if datacube is None:
            QMessageBox.information(self, "Probe Kernel", "Load a py4DSTEM DataCube first.")
            return
        roi = (
            self.roi_rx_start.value(),
            self.roi_rx_end.value(),
            self.roi_ry_start.value(),
            self.roi_ry_end.value(),
        )
        self._start_worker(
            lambda cb: self.service.prepare_probe_kernel(datacube, *roi),
            self._handle_probe_kernel_result,
            "Preparing vacuum-probe kernel...",
        )

    def run_current_pattern(self) -> None:
        datacube = self.datacube_provider()
        if datacube is None:
            QMessageBox.information(self, "Bragg Peaks", "Load a py4DSTEM DataCube first.")
            return
        rx, ry, params = self.rx_spin.value(), self.ry_spin.value(), self._params()
        self._start_worker(
            lambda cb: self.service.detect_peaks(datacube, rx, ry, params),
            self._handle_peak_result,
            "Bragg peak detection running...",
        )

    def run_full_braggvectors(self) -> None:
        datacube = self.datacube_provider()
        if datacube is None:
            QMessageBox.information(self, "BraggVectors", "Load a py4DSTEM DataCube first.")
            return
        params = self._params()
        self._start_worker(
            lambda cb: self.service.compute_braggvectors(datacube, params),
            self._handle_braggvectors_result,
            "Full BraggVectors calculation running...",
        )

    def run_selected_positions(self) -> None:
        datacube = self.datacube_provider()
        shape = self.shape_provider()
        if datacube is None or shape is None:
            QMessageBox.information(self, "Bragg Peaks", "Load a py4DSTEM DataCube first.")
            return
        import numpy as np

        if self.selection_mode.currentText().startswith("Deterministic"):
            positions = [
                (
                    shape[0] // 3 + (shape[0] // 3) * index // 6,
                    shape[1] // 3 + (shape[1] // 3) * index // 6,
                )
                for index in range(6)
            ]
        else:
            rng = np.random.default_rng(0)
            positions = [
                (int(rx), int(ry))
                for rx, ry in zip(
                    rng.integers(shape[0] // 3, max(2 * shape[0] // 3, shape[0] // 3 + 1), size=6),
                    rng.integers(shape[1] // 3, max(2 * shape[1] // 3, shape[1] // 3 + 1), size=6),
                )
            ]
        params = self._params()
        self._start_worker(
            lambda cb: self.service.detect_selected_positions(datacube, positions, params),
            self._handle_selected_result,
            "Checking selected scan positions...",
        )

    def _build_layout(self) -> None:
        probe_group = QGroupBox("1 Probe / Kernel Preparation")
        probe_layout = QFormLayout(probe_group)
        probe_layout.addRow("ROI rx start", self.roi_rx_start)
        probe_layout.addRow("ROI rx end", self.roi_rx_end)
        probe_layout.addRow("ROI ry start", self.roi_ry_start)
        probe_layout.addRow("ROI ry end", self.roi_ry_end)
        probe_layout.addRow("", self.pick_roi_button)
        probe_layout.addRow("", self.prepare_kernel_button)

        params_group = QGroupBox("2 Bragg Detection Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.addRow("preset", self.preset_combo)
        params_layout.addRow("", self.apply_preset_button)
        params_layout.addRow("minAbsoluteIntensity", self.min_abs_spin)
        params_layout.addRow("minRelativeIntensity", self.min_rel_spin)
        params_layout.addRow("minPeakSpacing", self.spacing_spin)
        params_layout.addRow("edgeBoundary", self.edge_spin)
        params_layout.addRow("maxNumPeaks", self.max_peaks_spin)
        params_layout.addRow("sigma_cc", self.sigma_spin)
        params_layout.addRow("sigma_dp", self.sigma_dp_spin)
        params_layout.addRow("corrPower", self.corr_power_spin)
        params_layout.addRow("upsample_factor", self.upsample_spin)
        params_layout.addRow("radial background", self.radial_background_check)
        params_layout.addRow("Gaussian fallback", self.gaussian_fallback_check)
        params_layout.addRow("subpixel", self.subpixel_combo)

        diagnostics_group = QGroupBox("3 Diagnostics")
        diagnostics_layout = QFormLayout(diagnostics_group)
        diagnostics_layout.addRow("rx", self.rx_spin)
        diagnostics_layout.addRow("ry", self.ry_spin)
        diagnostics_layout.addRow("six-points", self.selection_mode)
        diagnostics_layout.addRow("", self.run_current_button)
        diagnostics_layout.addRow("", self.run_selected_button)

        full_group = QGroupBox("4 Full BraggVectors")
        full_layout = QVBoxLayout(full_group)
        full_layout.addWidget(self.run_full_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(probe_group)
        left_layout.addWidget(params_group)
        left_layout.addWidget(diagnostics_group)
        left_layout.addWidget(full_group)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.count_label)
        left_layout.addWidget(self.quality_label)
        left_layout.addWidget(self.table)
        for button in [
            self.prepare_kernel_button,
            self.pick_roi_button,
            self.run_current_button,
            self.run_selected_button,
            self.run_full_button,
        ]:
            button.setMinimumHeight(30)

        left = QWidget()
        left.setLayout(left_layout)
        self.controls_panel = left

        layout = QHBoxLayout(self)
        layout.addWidget(self.workspace)

    def _float_input(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        unit: str = "",
        step: float = 1,
    ) -> NumericLineEdit:
        control = NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)
        control.setSingleStep(step)
        return control

    def _int_input(
        self,
        minimum: int,
        maximum: int,
        value: int,
        unit: str = "",
    ) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit, integer=True)

    def _params(self) -> BraggDetectionParams:
        return BraggDetectionParams(
            min_absolute_intensity=self.min_abs_spin.value(),
            min_relative_intensity=self.min_rel_spin.value(),
            min_peak_spacing=self.spacing_spin.value(),
            edge_boundary=self.edge_spin.value(),
            max_num_peaks=self.max_peaks_spin.value(),
            sigma_cc=self.sigma_spin.value(),
            sigma_dp=self.sigma_dp_spin.value(),
            corr_power=self.corr_power_spin.value(),
            upsample_factor=self.upsample_spin.value(),
            radial_background_subtraction=self.radial_background_check.isChecked(),
            subpixel=self.subpixel_combo.currentText(),
            allow_gaussian_fallback=self.gaussian_fallback_check.isChecked(),
            cuda=self.cuda_enabled,
        )

    def bragg_detection_params(self) -> BraggDetectionParams:
        return self._params()

    def apply_au_preset(self) -> None:
        preset = CBSPreset.au_notebook().bragg
        for control, value in [
            (self.min_abs_spin, preset.min_absolute_intensity),
            (self.min_rel_spin, preset.min_relative_intensity),
            (self.spacing_spin, preset.min_peak_spacing),
            (self.edge_spin, preset.edge_boundary),
            (self.max_peaks_spin, preset.max_num_peaks),
            (self.sigma_spin, preset.sigma_cc),
            (self.sigma_dp_spin, preset.sigma_dp),
            (self.corr_power_spin, preset.corr_power),
            (self.upsample_spin, preset.upsample_factor),
        ]:
            control.setValue(value)
        self.subpixel_combo.setCurrentText(preset.subpixel)
        self.radial_background_check.setChecked(preset.radial_background_subtraction)
        self.gaussian_fallback_check.setChecked(False)
        self.status_label.setText("Applied editable 01_CBS Au Bragg-detection preset.")

    def _start_worker(self, operation, finished_slot, status: str) -> None:
        all_buttons = (
            self.run_current_button, self.run_full_button, self.run_selected_button,
            self.prepare_kernel_button, self.pick_roi_button,
        )
        for button in all_buttons:
            button.setEnabled(False)
        self._pending_result_handler = finished_slot
        self._pending_status = status
        params = self._params()
        started = self._start_background(
            "Bragg calculation",
            operation,
            parameters={
                "minAbsoluteIntensity": params.min_absolute_intensity,
                "minRelativeIntensity": params.min_relative_intensity,
                "minPeakSpacing": params.min_peak_spacing,
                "edgeBoundary": params.edge_boundary,
                "maxNumPeaks": params.max_num_peaks,
                "template": "vacuum probe" if self.service.probe_kernel is not None else "explicit Gaussian fallback",
                "corrPower": params.corr_power,
                "sigma_dp": params.sigma_dp,
                "sigma_cc": params.sigma_cc,
                "upsample_factor": params.upsample_factor,
                "radial_background_subtraction": params.radial_background_subtraction,
                "CUDA": params.cuda,
            },
        )
        if not started:
            for button in all_buttons:
                button.setEnabled(True)

    def _on_start(self, name: str) -> None:
        status = getattr(self, "_pending_status", None) or f"Running {name}..."
        self.status_label.setText(status)
        self.log_panel.log(status)

    def _handle_result(self, result) -> None:
        handler = self._pending_result_handler
        self._pending_result_handler = None
        if handler is not None:
            handler(result)

    def _handle_peak_result(self, result: PeakDetectionResult) -> None:
        self.table.setHorizontalHeaderLabels(["index", "qx", "qy", "intensity", "distance"])
        self.viewer.set_image(result.diffraction_pattern)
        self.viewer.clear_points()
        if len(result.peaks):
            self.viewer.set_points(result.peaks[:, 0], result.peaks[:, 1])
        self.workspace.append_result(FigureResult(
            f"Single Position: {len(result.peaks)} peaks",
            result.diffraction_pattern,
            points=result.peaks,
        ))
        self._fill_table(result.peaks)
        self.count_label.setText(f"Peaks: {len(result.peaks)}")
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(f"Bragg peak detection completed: {len(result.peaks)} peaks.")
        self.log_panel.process_finished(
            "Bragg calculation", f"single position, {len(result.peaks)} peaks"
        )
        self.workflow_state.mark_completed(WorkflowStep.BRAGG_SINGLE)

    def _handle_braggvectors_result(self, result: BraggVectorsResult) -> None:
        count = "unknown" if result.peak_count is None else str(result.peak_count)
        self.status_label.setText(f"BraggVectors done in {result.elapsed_seconds:.2f} s")
        self.count_label.setText(f"BraggVectors peaks: {count}")
        quality = result.quality
        zero_fraction = float(getattr(quality, "zero_detection_fraction", np.mean(quality.failure_mask)))
        edge_clipped = int(getattr(quality, "edge_clipped_peak_count", 0))
        distribution = np.asarray(getattr(quality, "peak_count_distribution", quality.peak_count_map))
        warning = " Review parameters before accepting." if zero_fraction else ""
        self.quality_label.setText(
            "Quality: "
            f"zero-detection={zero_fraction:.1%}, "
            f"edge-clipped peaks={edge_clipped}, "
            f"median peaks/scan={np.nanmedian(distribution):.3g}.{warning}"
        )
        self.log_panel.log(f"Full BraggVectors completed: peaks={count}.")
        self.log_panel.process_finished("Bragg calculation", f"full map, peaks={count}")
        self.full_map_viewer.set_image(result.bragg_vector_map)
        self.workspace.append_result(FigureResult(
            "Full Bragg Vector Map",
            result.bragg_vector_map,
            viewer=self.full_map_viewer,
            bragg_sampling_provider=self._sampled_bragg_vector_map,
        ))

        self.braggvectors_ready.emit()
        self.workflow_state.mark_completed(WorkflowStep.BRAGG_FULL)
        if self.result_registry is not None:
            kernel = self.service.probe_kernel
            metadata = {
                "peak_count": result.peak_count,
                "zero_detection_fraction": zero_fraction,
                "edge_clipped_peak_count": edge_clipped,
                "kernel_shape": tuple(kernel.shape) if kernel is not None else None,
                "kernel_sum": float(np.sum(kernel)) if kernel is not None else None,
                **self.params_snapshot(),
            }
            self.result_registry.register(
                "bragg vector map",
                "Bragg disks",
                result.bragg_vector_map,
                ("npy", "png", "tiff"),
                metadata,
            )
            for name, image in [
                ("peak count map", result.quality.peak_count_map),
                ("mean peak intensity map", result.quality.mean_intensity_map),
                ("max peak intensity map", result.quality.max_intensity_map),
                ("detection failure mask", result.quality.failure_mask.astype(float)),
                ("peak count distribution", distribution),
                ("peak intensity distribution", np.asarray(getattr(result.quality, "intensity_distribution", []))),
            ]:
                self.result_registry.register(
                    name,
                    "Bragg disks",
                    image,
                    ("npy", "png", "tiff"),
                    metadata,
                )

    def _sampled_bragg_vector_map(self, sampling: int):
        braggvectors = self.service.braggvectors
        if braggvectors is None:
            raise ValueError("Run full BraggVectors first.")
        return np.asarray(braggvectors.histogram(mode="raw", sampling=sampling).data)

    def _handle_probe_kernel_result(self, result: ProbeKernelResult) -> None:
        self.status_label.setText(f"Probe kernel ready in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(
            "Vacuum-probe kernel prepared: "
            f"radius={result.probe_radius:.3g}, center=({result.center_x:.3g}, {result.center_y:.3g})."
        )
        self.log_panel.process_finished("Bragg calculation", "vacuum-probe kernel ready")
        self.workspace.append_results([
            FigureResult("Probe Kernel (R=24)", result.centered_kernel),
            FigureResult(
                "Probe Kernel Line Profiles (L=24, W=1)",
                result.profile_plot,
                image_kind="color",
                flip_x=True,
            ),
        ])
        self.workflow_state.mark_completed(WorkflowStep.PROBE_KERNEL)

    def _handle_selected_result(self, result: SelectedPeaksResult) -> None:
        self.table.setHorizontalHeaderLabels(["rx", "ry", "peak count"])
        self.table.setRowCount(len(result.positions))
        for row, ((rx, ry), count) in enumerate(zip(result.positions, result.peak_counts)):
            for col, value in enumerate((rx, ry, count)):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.count_label.setText(f"Selected-position peaks: {sum(result.peak_counts)}")
        self.status_label.setText(f"Selected positions done in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(f"Selected-position peak counts: {result.peak_counts}")
        self.log_panel.process_finished(
            "Bragg calculation", f"selected counts={result.peak_counts}"
        )
        self.workspace.append_results([
            FigureResult(
                f"({position[0]}, {position[1]}) | {count} peaks",
                pattern,
                points=peaks,
            )
            for position, pattern, peaks, count in zip(
                result.positions, result.patterns, result.peaks, result.peak_counts
            )
        ])
        self.workflow_state.mark_completed(WorkflowStep.BRAGG_SELECTED)

    def _handle_error(self, message: str) -> None:
        self._pending_result_handler = None
        self.status_label.setText("Failed")
        self.log_panel.log(f"Bragg operation failed: {message}")
        self.log_panel.process_failed("Bragg calculation", message)
        QMessageBox.warning(self, "Bragg Peaks", message)

    def _fill_table(self, peaks) -> None:
        self.table.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            qx, qy, intensity = peak[:3]
            distance = (float(qx) ** 2 + float(qy) ** 2) ** 0.5
            for col, value in enumerate((row, qx, qy, intensity, distance)):
                text = str(value) if col == 0 else f"{float(value):.4g}"
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _clear_worker_refs(self) -> None:
        super()._clear_worker_refs()
        self.run_current_button.setEnabled(True)
        self.run_full_button.setEnabled(True)
        self.run_selected_button.setEnabled(True)
        self.prepare_kernel_button.setEnabled(True)
        self.pick_roi_button.setEnabled(True)

    def _watch_parameters(self) -> None:
        all_detection_steps = [
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
        ]
        self.workflow_state.watch(self.rx_spin, WorkflowStep.BRAGG_SINGLE, "valueChanged")
        self.workflow_state.watch(self.ry_spin, WorkflowStep.BRAGG_SINGLE, "valueChanged")
        for spin in [
            self.min_abs_spin,
            self.min_rel_spin,
            self.spacing_spin,
            self.edge_spin,
            self.max_peaks_spin,
            self.sigma_spin,
            self.sigma_dp_spin,
            self.corr_power_spin,
            self.upsample_spin,
        ]:
            self.workflow_state.watch(spin, all_detection_steps, "valueChanged")
        self.workflow_state.watch(
            self.subpixel_combo, all_detection_steps, "currentTextChanged"
        )
        for control in [self.radial_background_check, self.gaussian_fallback_check]:
            self.workflow_state.watch(control, all_detection_steps, "toggled")
        for spin in [
            self.roi_rx_start,
            self.roi_rx_end,
            self.roi_ry_start,
            self.roi_ry_end,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.PROBE_KERNEL, "valueChanged")
            spin.valueChanged.connect(lambda _value: self._update_roi_overlay())

    def _refresh_stale_status(self) -> None:
        steps = [
            WorkflowStep.PROBE_KERNEL,
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
        ]
        if self.workflow_state.any_stale(steps):
            self.status_label.setText(STALE_RESULTS_MESSAGE)

    def _sync_virtual_image(self) -> None:
        image = self.virtual_image_provider()
        if image is not None:
            self.set_virtual_image(image)

    def start_roi_pick(self) -> None:
        if self.roi_viewer.raw_image is None:
            self._sync_virtual_image()
        if self.roi_viewer.raw_image is None:
            QMessageBox.information(
                self,
                "Probe ROI",
                "Run a bright-field or dark-field virtual image first.",
            )
            return
        self.roi_pick_points = []
        self.roi_viewer.set_interactive_roi_rect(
            self.roi_rx_start.value(),
            self.roi_rx_end.value(),
            self.roi_ry_start.value(),
            self.roi_ry_end.value(),
        )
        self.status_label.setText("Drag or resize the ROI box on the virtual image.")

    def _handle_roi_click(self, x: int, y: int) -> None:
        if not self.status_label.text().startswith("Click two corners") and not self.roi_pick_points:
            return
        self.roi_pick_points.append((x, y))
        if len(self.roi_pick_points) < 2:
            self.status_label.setText("First ROI corner selected. Click the opposite corner.")
            return
        (x1, y1), (x2, y2) = self.roi_pick_points[:2]
        rx_start, rx_end = sorted((x1, x2))
        ry_start, ry_end = sorted((y1, y2))
        self.roi_pick_points = []
        self.roi_rx_start.setValue(rx_start)
        self.roi_rx_end.setValue(rx_end + 1)
        self.roi_ry_start.setValue(ry_start)
        self.roi_ry_end.setValue(ry_end + 1)
        self._update_roi_overlay()
        self.status_label.setText(
            f"Probe ROI set from virtual image: rx={rx_start}:{rx_end + 1}, ry={ry_start}:{ry_end + 1}"
        )

    def _update_roi_overlay(self) -> None:
        if self.roi_viewer.raw_image is None:
            return
        self.roi_viewer.set_roi_rect(
            self.roi_rx_start.value(),
            self.roi_rx_end.value(),
            self.roi_ry_start.value(),
            self.roi_ry_end.value(),
        )

    def _handle_drawn_roi_changed(
        self,
        rx_start: int,
        rx_end: int,
        ry_start: int,
        ry_end: int,
    ) -> None:
        for spin, value in [
            (self.roi_rx_start, rx_start),
            (self.roi_rx_end, rx_end),
            (self.roi_ry_start, ry_start),
            (self.roi_ry_end, ry_end),
        ]:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.workflow_state.parameters_updated(WorkflowStep.PROBE_KERNEL)
        self.status_label.setText(
            f"Probe ROI set from drawn box: rx={rx_start}:{rx_end}, ry={ry_start}:{ry_end}"
        )

    def params_snapshot(self) -> dict[str, object]:
        params = self._params()
        return {
            "rx": self.rx_spin.value(),
            "ry": self.ry_spin.value(),
            "min_absolute_intensity": params.min_absolute_intensity,
            "min_relative_intensity": params.min_relative_intensity,
            "min_peak_spacing": params.min_peak_spacing,
            "edge_boundary": params.edge_boundary,
            "max_num_peaks": params.max_num_peaks,
            "sigma_cc": params.sigma_cc,
            "sigma_dp": params.sigma_dp,
            "corr_power": params.corr_power,
            "upsample_factor": params.upsample_factor,
            "radial_background_subtraction": params.radial_background_subtraction,
            "allow_gaussian_fallback": params.allow_gaussian_fallback,
            "subpixel": params.subpixel,
            "roi_rx_start": self.roi_rx_start.value(),
            "roi_rx_end": self.roi_rx_end.value(),
            "roi_ry_start": self.roi_ry_start.value(),
            "roi_ry_end": self.roi_ry_end.value(),
            "cuda": params.cuda,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        int_controls = {
            "rx": self.rx_spin,
            "ry": self.ry_spin,
            "min_peak_spacing": self.spacing_spin,
            "edge_boundary": self.edge_spin,
            "max_num_peaks": self.max_peaks_spin,
            "upsample_factor": self.upsample_spin,
            "roi_rx_start": self.roi_rx_start,
            "roi_rx_end": self.roi_rx_end,
            "roi_ry_start": self.roi_ry_start,
            "roi_ry_end": self.roi_ry_end,
        }
        float_controls = {
            "min_absolute_intensity": self.min_abs_spin,
            "min_relative_intensity": self.min_rel_spin,
            "sigma_cc": self.sigma_spin,
            "sigma_dp": self.sigma_dp_spin,
            "corr_power": self.corr_power_spin,
        }
        for key, spin in int_controls.items():
            if key in params:
                spin.setValue(int(params[key]))
        for key, spin in float_controls.items():
            if key in params:
                spin.setValue(float(params[key]))
        if "subpixel" in params:
            self.subpixel_combo.setCurrentText(str(params["subpixel"]))
        if "radial_background_subtraction" in params:
            self.radial_background_check.setChecked(bool(params["radial_background_subtraction"]))
        if "allow_gaussian_fallback" in params:
            self.gaussian_fallback_check.setChecked(bool(params["allow_gaussian_fallback"]))
