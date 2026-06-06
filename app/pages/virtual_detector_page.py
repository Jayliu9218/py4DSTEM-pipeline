from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services.virtual_detector_service import (
    VirtualDetectorParams,
    VirtualDetectorResult,
    VirtualDetectorService,
)
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class VirtualDetectorWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: VirtualDetectorService,
        source,
        params: VirtualDetectorParams,
    ) -> None:
        super().__init__()
        self.service = service
        self.source = source
        self.params = params

    def run(self) -> None:
        try:
            self.finished.emit(self.service.compute(self.source, self.params))
        except Exception as exc:
            self.failed.emit(str(exc))


class VirtualDetectorPage(QWidget):
    def __init__(
        self,
        source_provider: Callable[[], object | None],
        shape_provider: Callable[[], tuple[int, int, int, int] | None],
        log_panel: LogPanel,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.shape_provider = shape_provider
        self.log_panel = log_panel
        self.service = VirtualDetectorService()
        self.result: np.ndarray | None = None
        self.worker_thread: QThread | None = None
        self.worker: VirtualDetectorWorker | None = None

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                VirtualDetectorService.BRIGHT_FIELD,
                VirtualDetectorService.ANNULAR_DARK_FIELD,
                VirtualDetectorService.CUSTOM_ANNULAR,
            ]
        )

        self.center_x_spin = self._make_float_spin(0, 100000, 1)
        self.center_y_spin = self._make_float_spin(0, 100000, 1)
        self.inner_radius_spin = self._make_float_spin(0, 100000, 0)
        self.outer_radius_spin = self._make_float_spin(0.1, 100000, 10)

        self.run_button = QPushButton("Run")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.status_label = QLabel("Idle")
        self.viewer = ImageViewer()

        self.run_button.clicked.connect(self.run_detector)
        self.export_button.clicked.connect(self.export_result)
        self.mode_combo.currentTextChanged.connect(self._sync_mode_state)

        self._build_layout()
        self._sync_mode_state()

    def _build_layout(self) -> None:
        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow("Mode", self.mode_combo)
        form.addRow("center_x", self.center_x_spin)
        form.addRow("center_y", self.center_y_spin)
        form.addRow("inner_radius", self.inner_radius_spin)
        form.addRow("outer_radius", self.outer_radius_spin)

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.export_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(controls)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        left = QWidget()
        left.setLayout(left_layout)

        layout = QHBoxLayout(self)
        layout.addWidget(left, 0)
        layout.addWidget(self.viewer, 1)

    def _make_float_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        return spin

    def refresh_defaults_from_datacube(self) -> None:
        shape = self.shape_provider()
        if shape is None or len(shape) != 4:
            return

        qx, qy = shape[2], shape[3]
        self.center_x_spin.setValue((qx - 1) / 2)
        self.center_y_spin.setValue((qy - 1) / 2)
        self.inner_radius_spin.setValue(max(min(qx, qy) * 0.08, 1))
        self.outer_radius_spin.setValue(max(min(qx, qy) * 0.25, 2))
        self.log_panel.log("Virtual detector defaults updated from current DataCube.")

    def run_detector(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Virtual Detector", "Load a 4D DataCube first.")
            return

        params = self._params_from_ui()
        self.status_label.setText("Running...")
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.log_panel.log(f"Virtual detector started: {params.mode}")

        self.worker_thread = QThread()
        self.worker = VirtualDetectorWorker(self.service, source, params)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker_refs)
        self.worker_thread.start()

    def export_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Export", "No virtual image has been computed yet.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export virtual image",
            "",
            "PNG (*.png);;TIFF (*.tif *.tiff);;NumPy (*.npy)",
        )
        if not file_path:
            return

        try:
            self._write_result(Path(file_path), selected_filter)
            self.log_panel.log(f"Virtual detector result exported: {file_path}")
        except Exception as exc:
            self.log_panel.log(f"Export failed: {exc}")
            QMessageBox.warning(self, "Export failed", str(exc))

    def _params_from_ui(self) -> VirtualDetectorParams:
        return VirtualDetectorParams(
            mode=self.mode_combo.currentText(),
            center_x=self.center_x_spin.value(),
            center_y=self.center_y_spin.value(),
            inner_radius=self.inner_radius_spin.value(),
            outer_radius=self.outer_radius_spin.value(),
        )

    def _handle_finished(self, result: VirtualDetectorResult) -> None:
        self.result = result.image
        self.viewer.set_image(result.image)
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.log_panel.log(
            f"Virtual detector completed: {result.mode}, elapsed={result.elapsed_seconds:.2f} s"
        )

    def _handle_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(self.result is not None)
        self.log_panel.log(f"Virtual detector failed: {message}")
        QMessageBox.warning(self, "Virtual Detector", message)

    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _sync_mode_state(self) -> None:
        is_bf = self.mode_combo.currentText() == VirtualDetectorService.BRIGHT_FIELD
        self.inner_radius_spin.setEnabled(not is_bf)

    def _write_result(self, path: Path, selected_filter: str) -> None:
        suffix = path.suffix.lower()
        if "PNG" in selected_filter and suffix != ".png":
            path = path.with_suffix(".png")
        elif "TIFF" in selected_filter and suffix not in {".tif", ".tiff"}:
            path = path.with_suffix(".tif")
        elif "NumPy" in selected_filter and suffix != ".npy":
            path = path.with_suffix(".npy")

        if path.suffix.lower() == ".npy":
            np.save(path, self.result)
        elif path.suffix.lower() in {".tif", ".tiff"}:
            tifffile.imwrite(path, np.asarray(self.result))
        elif path.suffix.lower() == ".png":
            plt.imsave(path, np.asarray(self.result), cmap="gray")
        else:
            raise ValueError("Supported export formats are PNG, TIFF, and NPY.")
