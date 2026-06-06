from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
)
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class PeakDetectionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service: BraggStrainService, datacube, rx: int, ry: int, params: BraggDetectionParams) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.rx = rx
        self.ry = ry
        self.params = params

    def run(self) -> None:
        try:
            self.finished.emit(self.service.detect_peaks(self.datacube, self.rx, self.ry, self.params))
        except Exception as exc:
            self.failed.emit(str(exc))


class BraggVectorsWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service: BraggStrainService, datacube, params: BraggDetectionParams) -> None:
        super().__init__()
        self.service = service
        self.datacube = datacube
        self.params = params

    def run(self) -> None:
        try:
            self.finished.emit(self.service.compute_braggvectors(self.datacube, self.params))
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
        self.min_abs_spin = self._float_spin(0, 1e12, 0)
        self.min_rel_spin = self._float_spin(0, 1, 0.005, decimals=4, step=0.001)
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(1, 10000)
        self.spacing_spin.setValue(8)
        self.edge_spin = QSpinBox()
        self.edge_spin.setRange(0, 10000)
        self.edge_spin.setValue(5)
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 10000)
        self.max_peaks_spin.setValue(70)
        self.sigma_spin = self._float_spin(0.5, 1000, 2)

        self.run_current_button = QPushButton("Run Current Pattern")
        self.run_full_button = QPushButton("Run Full BraggVectors")
        self.status_label = QLabel("Idle")
        self.count_label = QLabel("Peaks: -")
        self.viewer = ImageViewer()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["qx", "qy", "intensity"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.run_current_button.clicked.connect(self.run_current_pattern)
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
        self.log_panel.log("Bragg Peaks controls updated from current DataCube.")

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

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_current_button)
        buttons.addWidget(self.run_full_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(controls)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.count_label)
        left_layout.addWidget(self.table)

        left = QWidget()
        left.setLayout(left_layout)

        layout = QHBoxLayout(self)
        layout.addWidget(left, 0)
        layout.addWidget(self.viewer, 1)

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
        )

    def _start_worker(self, worker: QObject, finished_slot, status: str) -> None:
        self.status_label.setText(status)
        self.run_current_button.setEnabled(False)
        self.run_full_button.setEnabled(False)
        self.log_panel.log(status)

        self.worker_thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.failed.connect(self._handle_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _handle_peak_result(self, result: PeakDetectionResult) -> None:
        self.viewer.set_image(result.diffraction_pattern)
        if len(result.peaks):
            self.viewer.set_points(result.peaks[:, 0], result.peaks[:, 1])
        self._fill_table(result.peaks)
        self.count_label.setText(f"Peaks: {len(result.peaks)}")
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.log_panel.log(f"Bragg peak detection completed: {len(result.peaks)} peaks.")

    def _handle_braggvectors_result(self, result: BraggVectorsResult) -> None:
        count = "unknown" if result.peak_count is None else str(result.peak_count)
        self.status_label.setText(f"BraggVectors done in {result.elapsed_seconds:.2f} s")
        self.count_label.setText(f"BraggVectors peaks: {count}")
        self.log_panel.log(f"Full BraggVectors completed: peaks={count}.")
        self.braggvectors_ready.emit()

    def _handle_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Bragg operation failed: {message}")
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
