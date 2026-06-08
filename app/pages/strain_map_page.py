from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import BraggStrainService, StrainMapParams, StrainMapResult
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.progress_stream import ProgressStream


class StrainMapWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, service: BraggStrainService, braggvectors, params: StrainMapParams) -> None:
        super().__init__()
        self.service = service
        self.braggvectors = braggvectors
        self.params = params

    def run(self) -> None:
        try:
            stream = ProgressStream(self.progress.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                self.finished.emit(self.service.compute_strain_map(self.braggvectors, self.params))
            stream.flush()
        except Exception as exc:
            self.failed.emit(str(exc))


class StrainMapPage(QWidget):
    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.service = service
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.result: StrainMapResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: StrainMapWorker | None = None
        self.roi_pick_points: list[tuple[int, int]] = []

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
        self.color_mode = QComboBox()
        self.color_mode.addItems(["auto symmetric", "percentile 1-99", "manual min/max"])
        self.color_min_spin = self._float_spin(-1e6, 1e6, -1)
        self.color_max_spin = self._float_spin(-1e6, 1e6, 1)
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
        self.pick_roi_button = QPushButton("Pick ROI From Map")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        self.viewer_tabs = QTabWidget()
        self.viewers = {
            name: ImageViewer()
            for name in [
                "exx",
                "eyy",
                "exy",
                "theta",
                "principal strain 1",
                "principal strain 2",
                "fit residual",
                "valid mask",
            ]
        }
        for name, viewer in self.viewers.items():
            self.viewer_tabs.addTab(viewer, name)
            viewer.image_clicked.connect(self._handle_roi_click)

        self.run_button.clicked.connect(self.run_strain_map)
        self.pick_roi_button.clicked.connect(self.start_roi_pick)
        self.export_button.clicked.connect(self.export_result)
        self.color_mode.currentTextChanged.connect(lambda _text: self._display_result())
        self.color_min_spin.valueChanged.connect(lambda _value: self._display_result())
        self.color_max_spin.valueChanged.connect(lambda _value: self._display_result())
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def notify_braggvectors_ready(self) -> None:
        self.status_label.setText("BraggVectors available")

    def run_strain_map(self) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            QMessageBox.information(self, "Strain Map", "Run full BraggVectors first.")
            return
        warning = self._calibration_warning(braggvectors)
        if warning:
            self.log_panel.log(f"WARN  {warning}")
            self.status_label.setText(warning)

        self.status_label.setText("Running...")
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.log_panel.log("Strain map calculation running...")
        self.log_panel.process_started(
            "StrainMap",
            f"reference={self.reference_mode.currentText()}, rotation={self.rotation_spin.value():g}",
        )
        self.log_panel.process_snapshot(
            ProcessSnapshot(
                step="Strain map",
                parameters={
                    "reference": self.reference_mode.currentText(),
                    "roi": (
                        self.roi_rx_start.value(),
                        self.roi_rx_end.value(),
                        self.roi_ry_start.value(),
                        self.roi_ry_end.value(),
                    ),
                    "rotation": self.rotation_spin.value(),
                    "max_peak_spacing": self.max_spacing_spin.value(),
                },
            )
        )

        self.worker_thread = QThread()
        self.worker = StrainMapWorker(self.service, braggvectors, self._params())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.progress.connect(self.log_panel.process_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _calibration_warning(self, braggvectors) -> str:
        calstate = getattr(braggvectors, "calstate", {})
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
            "Calibration is incomplete; strain will continue, but accuracy may be lower. "
            f"Missing/applied-off corrections: {', '.join(missing)}."
        )

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
        form.addRow("reference mode", self.reference_mode)
        form.addRow("color range", self.color_mode)
        form.addRow("manual color min", self.color_min_spin)
        form.addRow("manual color max", self.color_max_spin)
        form.addRow("reference ROI rx start", self.roi_rx_start)
        form.addRow("reference ROI rx end", self.roi_rx_end)
        form.addRow("reference ROI ry start", self.roi_ry_start)
        form.addRow("reference ROI ry end", self.roi_ry_end)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(controls)
        left_layout.addWidget(self.pick_roi_button)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.export_button)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        self.controls_panel = left
        layout = QHBoxLayout(self)
        layout.addWidget(self.viewer_tabs)

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
        self._display_result()
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.export_button.setEnabled(True)
        self.log_panel.log(f"Strain map completed in {result.elapsed_seconds:.2f} s.")
        self.log_panel.process_finished("StrainMap", f"elapsed={result.elapsed_seconds:.2f} s")
        self.workflow_state.mark_completed(WorkflowStep.STRAIN_MAP)
        if self.result_registry is not None:
            metadata = self.params_snapshot()
            self.result_registry.register(
                "strain map components",
                "strain",
                result.components,
                ("npz",),
                metadata,
            )
            for name, image in result.components.items():
                self.result_registry.register(
                    name,
                    "strain",
                    image,
                    ("npy", "png", "tiff"),
                    metadata,
                )

    def _handle_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.export_button.setEnabled(self.result is not None)
        self.log_panel.log(f"Strain map failed: {message}")
        self.log_panel.process_failed("StrainMap", message)
        QMessageBox.warning(self, "Strain Map", message)

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.run_button.setEnabled(True)

    def start_roi_pick(self) -> None:
        self.roi_pick_points = []
        self.status_label.setText("Click two corners on any strain result image to set reference ROI.")

    def _handle_roi_click(self, x: int, y: int) -> None:
        if self.roi_pick_points or self.status_label.text().startswith("Click two corners"):
            self.roi_pick_points.append((x, y))
            if len(self.roi_pick_points) < 2:
                self.status_label.setText("First ROI corner selected. Click the opposite corner.")
                return
            (x1, y1), (x2, y2) = self.roi_pick_points[:2]
            rx_start, rx_end = sorted((x1, x2))
            ry_start, ry_end = sorted((y1, y2))
            self.roi_rx_start.setValue(rx_start)
            self.roi_rx_end.setValue(rx_end + 1)
            self.roi_ry_start.setValue(ry_start)
            self.roi_ry_end.setValue(ry_end + 1)
            self.roi_pick_points = []
            self.status_label.setText(
                f"Reference ROI set: rx={rx_start}:{rx_end + 1}, ry={ry_start}:{ry_end + 1}"
            )

    def _display_result(self) -> None:
        if self.result is None:
            return
        for name, viewer in self.viewers.items():
            image = self.result.components.get(name)
            if image is None:
                viewer.clear(f"{name} is not available for this result.")
            else:
                viewer.set_image(image, levels=self._levels_for(image))

    def _levels_for(self, image) -> tuple[float, float] | None:
        array = np.asarray(image, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None
        mode = self.color_mode.currentText()
        if mode == "manual min/max":
            return (self.color_min_spin.value(), self.color_max_spin.value())
        if mode == "percentile 1-99":
            return (float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99)))
        limit = float(np.nanmax(np.abs(finite)))
        if limit == 0:
            return None
        return (-limit, limit)

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

    def _watch_parameters(self) -> None:
        for spin in [
            self.rotation_spin,
            self.max_spacing_spin,
            self.min_abs_spin,
            self.min_rel_spin,
            self.min_spacing_spin,
            self.edge_spin,
            self.max_peaks_spin,
            self.roi_rx_start,
            self.roi_rx_end,
            self.roi_ry_start,
            self.roi_ry_end,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.STRAIN_MAP, "valueChanged")
        self.workflow_state.watch(
            self.reference_mode, WorkflowStep.STRAIN_MAP, "currentTextChanged"
        )

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.is_stale(WorkflowStep.STRAIN_MAP):
            self.status_label.setText(STALE_RESULTS_MESSAGE)

    def params_snapshot(self) -> dict[str, object]:
        params = self._params()
        return {
            "coordinate_rotation": params.coordinate_rotation,
            "max_peak_spacing": params.max_peak_spacing,
            "min_absolute_intensity": params.min_absolute_intensity,
            "min_relative_intensity": params.min_relative_intensity,
            "min_spacing": params.min_spacing,
            "edge_boundary": params.edge_boundary,
            "max_num_peaks": params.max_num_peaks,
            "reference_mode": params.reference_mode,
            "roi_rx_start": params.roi_rx_start,
            "roi_rx_end": params.roi_rx_end,
            "roi_ry_start": params.roi_ry_start,
            "roi_ry_end": params.roi_ry_end,
            "color_mode": self.color_mode.currentText(),
            "color_min": self.color_min_spin.value(),
            "color_max": self.color_max_spin.value(),
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        float_controls = {
            "coordinate_rotation": self.rotation_spin,
            "max_peak_spacing": self.max_spacing_spin,
            "min_absolute_intensity": self.min_abs_spin,
            "min_relative_intensity": self.min_rel_spin,
            "min_spacing": self.min_spacing_spin,
            "color_min": self.color_min_spin,
            "color_max": self.color_max_spin,
        }
        int_controls = {
            "edge_boundary": self.edge_spin,
            "max_num_peaks": self.max_peaks_spin,
            "roi_rx_start": self.roi_rx_start,
            "roi_rx_end": self.roi_rx_end,
            "roi_ry_start": self.roi_ry_start,
            "roi_ry_end": self.roi_ry_end,
        }
        for key, spin in float_controls.items():
            if key in params:
                spin.setValue(float(params[key]))
        for key, spin in int_controls.items():
            if key in params:
                spin.setValue(int(params[key]))
        if "reference_mode" in params:
            self.reference_mode.setCurrentText(str(params["reference_mode"]))
        if "color_mode" in params:
            self.color_mode.setCurrentText(str(params["color_mode"]))
