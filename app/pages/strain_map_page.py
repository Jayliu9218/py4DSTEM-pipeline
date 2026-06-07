from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import BraggStrainService, StrainMapParams, StrainMapResult
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class StrainMapWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service: BraggStrainService, braggvectors, params: StrainMapParams) -> None:
        super().__init__()
        self.service = service
        self.braggvectors = braggvectors
        self.params = params

    def run(self) -> None:
        try:
            self.finished.emit(self.service.compute_strain_map(self.braggvectors, self.params))
        except Exception as exc:
            self.failed.emit(str(exc))


class StrainMapPage(QWidget):
    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.service = service
        self.log_panel = log_panel
        self.result: StrainMapResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: StrainMapWorker | None = None

        self.rotation_spin = self._float_spin(-360, 360, -21.5)
        self.max_spacing_spin = self._float_spin(0.1, 1000, 3)
        self.min_abs_spin = self._float_spin(0, 1e12, 1200)
        self.min_rel_spin = self._float_spin(0, 1, 0, decimals=4, step=0.001)
        self.min_spacing_spin = self._float_spin(0, 1000, 2)
        self.edge_spin = QSpinBox()
        self.edge_spin.setRange(0, 10000)
        self.edge_spin.setValue(1)
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 10000)
        self.max_peaks_spin.setValue(150)
        self.reference_mode = QComboBox()
        self.reference_mode.addItems(["roi_vectors", "auto_valid", "roi_mask"])
        self.roi_rx_start = QSpinBox()
        self.roi_rx_end = QSpinBox()
        self.roi_ry_start = QSpinBox()
        self.roi_ry_end = QSpinBox()
        for spin in [self.roi_rx_start, self.roi_rx_end, self.roi_ry_start, self.roi_ry_end]:
            spin.setRange(0, 100000)
        self.roi_rx_start.setValue(34)
        self.roi_rx_end.setValue(42)
        self.roi_ry_start.setValue(8)
        self.roi_ry_end.setValue(16)

        self.run_button = QPushButton("Run Strain Map")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.status_label = QLabel("Idle")

        self.viewer_tabs = QTabWidget()
        self.viewers = {name: ImageViewer() for name in ["exx", "eyy", "exy", "theta"]}
        for name, viewer in self.viewers.items():
            self.viewer_tabs.addTab(viewer, name)

        self.run_button.clicked.connect(self.run_strain_map)
        self.export_button.clicked.connect(self.export_result)
        self._build_layout()

    def notify_braggvectors_ready(self) -> None:
        self.status_label.setText("BraggVectors available")

    def run_strain_map(self) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            QMessageBox.information(self, "Strain Map", "Run full BraggVectors first.")
            return

        self.status_label.setText("Running...")
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.log_panel.log("Strain map calculation running...")

        self.worker_thread = QThread()
        self.worker = StrainMapWorker(self.service, braggvectors, self._params())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def export_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Export", "No strain map has been computed yet.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export strain map",
            "",
            "PNG summary (*.png);;TIFF stack (*.tif *.tiff);;NumPy stack (*.npy);;NumPy archive (*.npz)",
        )
        if not file_path:
            return

        path = self._path_with_filter_suffix(Path(file_path), selected_filter)
        try:
            self.service.export_strain_result(self.result, path)
            self.log_panel.log(f"Strain map exported: {path}")
        except Exception as exc:
            self.log_panel.log(f"Strain map export failed: {exc}")
            QMessageBox.warning(self, "Export failed", str(exc))

    def _build_layout(self) -> None:
        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow("coordinate_rotation", self.rotation_spin)
        form.addRow("max_peak_spacing", self.max_spacing_spin)
        form.addRow("minAbsoluteIntensity", self.min_abs_spin)
        form.addRow("minRelativeIntensity", self.min_rel_spin)
        form.addRow("minSpacing", self.min_spacing_spin)
        form.addRow("edgeBoundary", self.edge_spin)
        form.addRow("maxNumPeaks", self.max_peaks_spin)
        form.addRow("8.1 reference mode", self.reference_mode)
        form.addRow("reference ROI rx start", self.roi_rx_start)
        form.addRow("reference ROI rx end", self.roi_rx_end)
        form.addRow("reference ROI ry start", self.roi_ry_start)
        form.addRow("reference ROI ry end", self.roi_ry_end)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(self.run_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.viewer_tabs, 1)

    def _float_spin(self, minimum: float, maximum: float, value: float, decimals: int = 2, step: float = 1) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _params(self) -> StrainMapParams:
        return StrainMapParams(
            coordinate_rotation=self.rotation_spin.value(),
            max_peak_spacing=self.max_spacing_spin.value(),
            min_absolute_intensity=self.min_abs_spin.value(),
            min_relative_intensity=self.min_rel_spin.value(),
            min_spacing=self.min_spacing_spin.value(),
            edge_boundary=self.edge_spin.value(),
            max_num_peaks=self.max_peaks_spin.value(),
            reference_mode=self.reference_mode.currentText(),
            roi_rx_start=self.roi_rx_start.value(),
            roi_rx_end=self.roi_rx_end.value(),
            roi_ry_start=self.roi_ry_start.value(),
            roi_ry_end=self.roi_ry_end.value(),
        )

    def _handle_finished(self, result: StrainMapResult) -> None:
        self.result = result
        for name, image in result.components.items():
            self.viewers[name].set_image(image)
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.export_button.setEnabled(True)
        self.log_panel.log(f"Strain map completed in {result.elapsed_seconds:.2f} s.")

    def _handle_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.export_button.setEnabled(self.result is not None)
        self.log_panel.log(f"Strain map failed: {message}")
        QMessageBox.warning(self, "Strain Map", message)

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.run_button.setEnabled(True)

    def _path_with_filter_suffix(self, path: Path, selected_filter: str) -> Path:
        if "PNG" in selected_filter and path.suffix.lower() != ".png":
            return path.with_suffix(".png")
        if "TIFF" in selected_filter and path.suffix.lower() not in {".tif", ".tiff"}:
            return path.with_suffix(".tif")
        if "NumPy stack" in selected_filter and path.suffix.lower() != ".npy":
            return path.with_suffix(".npy")
        if "NumPy archive" in selected_filter and path.suffix.lower() != ".npz":
            return path.with_suffix(".npz")
        return path
