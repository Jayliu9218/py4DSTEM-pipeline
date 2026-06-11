from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
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
from app.widgets.progress_stream import ProgressStream


def _run_with_progress(worker: QObject, operation) -> None:
    stream = ProgressStream(worker.progress.emit)
    with redirect_stdout(stream), redirect_stderr(stream):
        operation()
    stream.flush()


class PeakDetectionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, service: BraggStrainService, datacube, rx: int, ry: int, params: BraggDetectionParams) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.rx = rx
        self.ry = ry
        self.params = params

    def run(self) -> None:
        try:
            _run_with_progress(
                self,
                lambda: self.finished.emit(
                    self.service.detect_peaks(self.datacube, self.rx, self.ry, self.params)
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BraggVectorsWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, service: BraggStrainService, datacube, params: BraggDetectionParams) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.params = params

    def run(self) -> None:
        try:
            _run_with_progress(
                self,
                lambda: self.finished.emit(self.service.compute_braggvectors(self.datacube, self.params)),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ProbeKernelWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, service: BraggStrainService, datacube, roi: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.roi = roi

    def run(self) -> None:
        try:
            _run_with_progress(
                self,
                lambda: self.finished.emit(self.service.prepare_probe_kernel(self.datacube, *self.roi)),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class SelectedPeaksWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, service, datacube, positions, params) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.positions = positions
        self.params = params

    def run(self) -> None:
        try:
            _run_with_progress(
                self,
                lambda: self.finished.emit(
                    self.service.detect_selected_positions(self.datacube, self.positions, self.params)
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BraggPeaksPage(QWidget):
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
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self.roi_pick_points: list[tuple[int, int]] = []

        self.rx_spin = self._int_input(0, 100000, 0, unit="px")
        self.ry_spin = self._int_input(0, 100000, 0, unit="px")
        self.min_abs_spin = self._float_input(0, 1e12, 2, unit="int.")
        self.min_rel_spin = self._float_input(0, 1, 0, decimals=4, unit="ratio")
        self.spacing_spin = self._int_input(1, 10000, 18, unit="px")
        self.edge_spin = self._int_input(0, 10000, 2, unit="px")
        self.max_peaks_spin = self._int_input(1, 10000, 100, unit="peaks")
        self.sigma_spin = self._float_input(0.5, 1000, 2, unit="px")
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
            ProbeKernelWorker(self.service, datacube, roi),
            self._handle_probe_kernel_result,
            "Preparing vacuum-probe kernel...",
        )

    def run_current_pattern(self) -> None:
        datacube = self.datacube_provider()
        if datacube is None:
            QMessageBox.information(self, "Bragg Peaks", "Load a py4DSTEM DataCube first.")
            return
        self._start_worker(
            PeakDetectionWorker(
                self.service,
                datacube,
                self.rx_spin.value(),
                self.ry_spin.value(),
                self._params(),
            ),
            self._handle_peak_result,
            "Bragg peak detection running...",
        )

    def run_full_braggvectors(self) -> None:
        datacube = self.datacube_provider()
        if datacube is None:
            QMessageBox.information(self, "BraggVectors", "Load a py4DSTEM DataCube first.")
            return
        self._start_worker(
            BraggVectorsWorker(self.service, datacube, self._params()),
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

        rng = np.random.default_rng(0)
        positions = [
            (int(rx), int(ry))
            for rx, ry in zip(
                rng.integers(shape[0] // 3, max(2 * shape[0] // 3, shape[0] // 3 + 1), size=6),
                rng.integers(shape[1] // 3, max(2 * shape[1] // 3, shape[1] // 3 + 1), size=6),
            )
        ]
        self._start_worker(
            SelectedPeaksWorker(self.service, datacube, positions, self._params()),
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
        params_layout.addRow("minAbsoluteIntensity", self.min_abs_spin)
        params_layout.addRow("minRelativeIntensity", self.min_rel_spin)
        params_layout.addRow("minPeakSpacing", self.spacing_spin)
        params_layout.addRow("edgeBoundary", self.edge_spin)
        params_layout.addRow("maxNumPeaks", self.max_peaks_spin)
        params_layout.addRow("template sigma", self.sigma_spin)
        params_layout.addRow("subpixel", self.subpixel_combo)

        diagnostics_group = QGroupBox("3 Diagnostics")
        diagnostics_layout = QFormLayout(diagnostics_group)
        diagnostics_layout.addRow("rx", self.rx_spin)
        diagnostics_layout.addRow("ry", self.ry_spin)
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
            template_sigma=self.sigma_spin.value(),
            subpixel=self.subpixel_combo.currentText(),
            cuda=self.cuda_enabled,
        )

    def bragg_detection_params(self) -> BraggDetectionParams:
        return self._params()

    def _start_worker(self, worker: QObject, finished_slot, status: str) -> None:
        self.status_label.setText(status)
        self.run_current_button.setEnabled(False)
        self.run_full_button.setEnabled(False)
        self.run_selected_button.setEnabled(False)
        self.prepare_kernel_button.setEnabled(False)
        self.pick_roi_button.setEnabled(False)
        self.log_panel.log(status)
        self.log_panel.process_started("Bragg calculation", status)
        params = self._params()
        self.log_panel.process_snapshot(
            ProcessSnapshot(
                step="Bragg disk detection",
                parameters={
                    "minRelativeIntensity": params.min_relative_intensity,
                    "minPeakSpacing": params.min_peak_spacing,
                    "edgeBoundary": params.edge_boundary,
                    "maxNumPeaks": params.max_num_peaks,
                    "template": "vacuum probe" if self.service.probe_kernel is not None else "gaussian",
                    "CUDA": params.cuda,
                },
            )
        )

        self.worker_thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.failed.connect(self._handle_failed)
        self.worker.progress.connect(self.log_panel.process_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

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
        self.log_panel.log(f"Full BraggVectors completed: peaks={count}.")
        self.log_panel.process_finished("Bragg calculation", f"full map, peaks={count}")
        self.full_map_viewer.set_image(result.bragg_vector_map)
        self.workspace.update_result(
            "full-map",
            FigureResult(
                "Full Bragg Vector Map",
                result.bragg_vector_map,
                viewer=self.full_map_viewer,
                bragg_sampling_provider=self._sampled_bragg_vector_map,
            ),
        )
        for name, image in [
            ("Peak Count Map", result.quality.peak_count_map),
            ("Mean Peak Intensity Map", result.quality.mean_intensity_map),
            ("Max Peak Intensity Map", result.quality.max_intensity_map),
            ("Detection Failure Mask", result.quality.failure_mask.astype(float)),
        ]:
            self.workspace.append_result(FigureResult(name, image))
        self.braggvectors_ready.emit()
        self.workflow_state.mark_completed(WorkflowStep.BRAGG_FULL)
        if self.result_registry is not None:
            metadata = {"peak_count": result.peak_count, **self.params_snapshot()}
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
        for position, pattern, peaks, count in zip(
            result.positions, result.patterns, result.peaks, result.peak_counts
        ):
            self.workspace.append_result(FigureResult(
                f"({position[0]}, {position[1]}) | {count} peaks",
                pattern,
                points=peaks,
            ))
        self.workflow_state.mark_completed(WorkflowStep.BRAGG_SELECTED)

    def _handle_failed(self, message: str) -> None:
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

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
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
        ]:
            self.workflow_state.watch(spin, all_detection_steps, "valueChanged")
        self.workflow_state.watch(
            self.subpixel_combo, all_detection_steps, "currentTextChanged"
        )
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
            "template_sigma": params.template_sigma,
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
            "roi_rx_start": self.roi_rx_start,
            "roi_rx_end": self.roi_rx_end,
            "roi_ry_start": self.roi_ry_start,
            "roi_ry_end": self.roi_ry_end,
        }
        float_controls = {
            "min_absolute_intensity": self.min_abs_spin,
            "min_relative_intensity": self.min_rel_spin,
            "template_sigma": self.sigma_spin,
        }
        for key, spin in int_controls.items():
            if key in params:
                spin.setValue(int(params[key]))
        for key, spin in float_controls.items():
            if key in params:
                spin.setValue(float(params[key]))
        if "subpixel" in params:
            self.subpixel_combo.setCurrentText(str(params["subpixel"]))
