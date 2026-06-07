from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
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
from app.widgets.image_viewer import ImageViewer
from app.widgets.image_grid_viewer import ImageGridViewer
from app.widgets.log_panel import LogPanel
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
        service: BraggStrainService,
        log_panel: LogPanel,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.shape_provider = shape_provider
        self.service = service
        self.log_panel = log_panel
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None

        self.rx_spin = QSpinBox()
        self.ry_spin = QSpinBox()
        self.min_abs_spin = self._float_spin(0, 1e12, 2)
        self.min_rel_spin = self._float_spin(0, 1, 0, decimals=4, step=0.001)
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(1, 10000)
        self.spacing_spin.setValue(18)
        self.edge_spin = QSpinBox()
        self.edge_spin.setRange(0, 10000)
        self.edge_spin.setValue(2)
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 10000)
        self.max_peaks_spin.setValue(100)
        self.sigma_spin = self._float_spin(0.5, 1000, 2)
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["poly", "multicorr", "pixel"])
        self.roi_rx_start = QSpinBox()
        self.roi_rx_end = QSpinBox()
        self.roi_ry_start = QSpinBox()
        self.roi_ry_end = QSpinBox()

        self.prepare_kernel_button = QPushButton("Prepare Vacuum-Probe Kernel")
        self.run_current_button = QPushButton("Run Current Pattern")
        self.run_selected_button = QPushButton("Check 6 Selected Positions")
        self.run_full_button = QPushButton("Run Full BraggVectors")
        self.status_label = QLabel("Idle")
        self.count_label = QLabel("Peaks: -")
        self.viewer = ImageViewer()
        self.selected_grid = ImageGridViewer()
        self.full_map_viewer = ImageViewer()
        self.visual_tabs = QTabWidget()
        self.visual_tabs.addTab(self.viewer, "Single Position")
        self.visual_tabs.addTab(self.selected_grid, "Selected 6 Positions")
        self.visual_tabs.addTab(self.full_map_viewer, "Full Bragg Vector Map")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["qx", "qy", "intensity"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.prepare_kernel_button.clicked.connect(self.prepare_probe_kernel)
        self.run_current_button.clicked.connect(self.run_current_pattern)
        self.run_selected_button.clicked.connect(self.run_selected_positions)
        self.run_full_button.clicked.connect(self.run_full_braggvectors)
        self._build_layout()

    def refresh_from_datacube(self) -> None:
        shape = self.shape_provider()
        if shape is None:
            return
        self.rx_spin.setMaximum(max(shape[0] - 1, 0))
        self.ry_spin.setMaximum(max(shape[1] - 1, 0))
        self.rx_spin.setValue(0)
        self.ry_spin.setValue(0)
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
        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow("rx", self.rx_spin)
        form.addRow("ry", self.ry_spin)
        form.addRow("minAbsoluteIntensity", self.min_abs_spin)
        form.addRow("minRelativeIntensity", self.min_rel_spin)
        form.addRow("minPeakSpacing", self.spacing_spin)
        form.addRow("edgeBoundary", self.edge_spin)
        form.addRow("maxNumPeaks", self.max_peaks_spin)
        form.addRow("template sigma", self.sigma_spin)
        form.addRow("subpixel", self.subpixel_combo)
        form.addRow("vacuum ROI rx start", self.roi_rx_start)
        form.addRow("vacuum ROI rx end", self.roi_rx_end)
        form.addRow("vacuum ROI ry start", self.roi_ry_start)
        form.addRow("vacuum ROI ry end", self.roi_ry_end)

        buttons = QHBoxLayout()
        buttons.addWidget(self.prepare_kernel_button)
        buttons.addWidget(self.run_current_button)
        buttons.addWidget(self.run_selected_button)
        buttons.addWidget(self.run_full_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(controls)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.count_label)
        left_layout.addWidget(self.table)

        left = QWidget()
        left.setLayout(left_layout)

        left.setFixedWidth(430)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.visual_tabs)
        splitter.setSizes([430, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def _float_spin(self, minimum: float, maximum: float, value: float, decimals: int = 2, step: float = 1) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _params(self) -> BraggDetectionParams:
        return BraggDetectionParams(
            min_absolute_intensity=self.min_abs_spin.value(),
            min_relative_intensity=self.min_rel_spin.value(),
            min_peak_spacing=self.spacing_spin.value(),
            edge_boundary=self.edge_spin.value(),
            max_num_peaks=self.max_peaks_spin.value(),
            template_sigma=self.sigma_spin.value(),
            subpixel=self.subpixel_combo.currentText(),
        )

    def _start_worker(self, worker: QObject, finished_slot, status: str) -> None:
        self.status_label.setText(status)
        self.run_current_button.setEnabled(False)
        self.run_full_button.setEnabled(False)
        self.run_selected_button.setEnabled(False)
        self.prepare_kernel_button.setEnabled(False)
        self.log_panel.log(status)
        self.log_panel.process_started("Bragg calculation", status)

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
        self.table.setHorizontalHeaderLabels(["qx", "qy", "intensity"])
        self.viewer.set_image(result.diffraction_pattern)
        self.viewer.clear_points()
        if len(result.peaks):
            self.viewer.set_points(result.peaks[:, 0], result.peaks[:, 1])
        self._fill_table(result.peaks)
        self.count_label.setText(f"Peaks: {len(result.peaks)}")
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(f"Bragg peak detection completed: {len(result.peaks)} peaks.")
        self.log_panel.process_finished(
            "Bragg calculation", f"single position, {len(result.peaks)} peaks"
        )
        self.visual_tabs.setCurrentWidget(self.viewer)

    def _handle_braggvectors_result(self, result: BraggVectorsResult) -> None:
        count = "unknown" if result.peak_count is None else str(result.peak_count)
        self.status_label.setText(f"BraggVectors done in {result.elapsed_seconds:.2f} s")
        self.count_label.setText(f"BraggVectors peaks: {count}")
        self.log_panel.log(f"Full BraggVectors completed: peaks={count}.")
        self.log_panel.process_finished("Bragg calculation", f"full map, peaks={count}")
        self.full_map_viewer.set_image(result.bragg_vector_map)
        self.visual_tabs.setCurrentWidget(self.full_map_viewer)
        self.braggvectors_ready.emit()

    def _handle_probe_kernel_result(self, result: ProbeKernelResult) -> None:
        self.status_label.setText(f"Probe kernel ready in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(
            "Vacuum-probe kernel prepared: "
            f"radius={result.probe_radius:.3g}, center=({result.center_x:.3g}, {result.center_y:.3g})."
        )
        self.log_panel.process_finished("Bragg calculation", "vacuum-probe kernel ready")

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
        self.selected_grid.clear()
        for index, (position, pattern, peaks, count) in enumerate(
            zip(result.positions, result.patterns, result.peaks, result.peak_counts)
        ):
            self.selected_grid.set_result(
                index,
                f"({position[0]}, {position[1]}) | {count} peaks",
                pattern,
                peaks,
            )
        self.visual_tabs.setCurrentWidget(self.selected_grid)

    def _handle_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Bragg operation failed: {message}")
        self.log_panel.process_failed("Bragg calculation", message)
        QMessageBox.warning(self, "Bragg Peaks", message)

    def _fill_table(self, peaks) -> None:
        self.table.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            for col, value in enumerate(peak[:3]):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))
        self.table.resizeColumnsToContents()

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.run_current_button.setEnabled(True)
        self.run_full_button.setEnabled(True)
        self.run_selected_button.setEnabled(True)
        self.prepare_kernel_button.setEnabled(True)
