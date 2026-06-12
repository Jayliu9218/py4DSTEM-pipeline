from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.services.bragg_strain_service import (
    BasisSelectionParams,
    BraggStrainService,
    StrainMapParams,
    StrainMapResult,
    StrainStageResult,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
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
                if self.service.strainmap is not None and hasattr(self.service.strainmap, "g1g2_map"):
                    self.finished.emit(self.service.calculate_strain_from_stages(self.braggvectors, self.params))
                else:
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

        self.rotation_spin = self._float_input(-360, 360, -21.5, unit="deg")
        self.max_spacing_spin = self._float_input(0.1, 1000, 3, unit="px")
        self.min_abs_spin = self._float_input(0, 1e12, 1200, unit="int.")
        self.min_rel_spin = self._float_input(0, 1, 0, decimals=4, unit="ratio")
        self.min_spacing_spin = self._float_input(0, 1000, 2, unit="px")
        self.edge_spin = self._int_input(0, 10000, 1, unit="px")
        self.max_peaks_spin = self._int_input(1, 10000, 150, unit="peaks")
        self.reference_mode = QComboBox()
        self.reference_mode.addItems(["auto_valid", "roi_vectors", "roi_mask", "manual_g1g2"])
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Final Strain","Process"])
        self.color_mode = QComboBox()
        self.color_mode.addItems(["auto symmetric", "percentile 1-99", "manual min/max"])
        self.color_min_spin = self._float_input(-1e6, 1e6, -1, unit="value")
        self.color_max_spin = self._float_input(-1e6, 1e6, 1, unit="value")
        self.roi_rx_start = self._int_input(0, 100000, 34, unit="px")
        self.roi_rx_end = self._int_input(0, 100000, 42, unit="px")
        self.roi_ry_start = self._int_input(0, 100000, 8, unit="px")
        self.roi_ry_end = self._int_input(0, 100000, 16, unit="px")
        self.manual_g1_x = self._float_input(-100000, 100000, 1)
        self.manual_g1_y = self._float_input(-100000, 100000, 0)
        self.manual_g2_x = self._float_input(-100000, 100000, 0)
        self.manual_g2_y = self._float_input(-100000, 100000, 1)
        self.index_origin = self._int_input(-1, 10000, -1)
        self.index_g1 = self._int_input(-1, 10000, -1)
        self.index_g2 = self._int_input(-1, 10000, -1)

        self.choose_basis_button = QPushButton("1 Choose Basis Vectors")
        self.accept_basis_button = QPushButton("2 Accept Basis Selection")
        self.set_spacing_button = QPushButton("3 Set Peak Spacing")
        self.fit_basis_button = QPushButton("4 Fit Basis Vectors")
        self.accept_fit_button = QPushButton("5 Accept Basis Fit")
        self.accept_reference_button = QPushButton("6 Review && Accept Reference")
        self.run_button = QPushButton("Run Strain Map")
        self.run_button.setEnabled(False)
        self.pick_roi_button = QPushButton("Pick ROI From Map")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.applied_qr_rotation_label = QLabel("missing / not applied")
        self.applied_qr_rotation_label.setWordWrap(True)

        self.workspace = AdaptiveImageWorkspace()

        self.run_button.clicked.connect(self.run_strain_map)
        self.choose_basis_button.clicked.connect(self.choose_basis_vectors)
        self.accept_basis_button.clicked.connect(lambda: self.accept_stage("basis_selection"))
        self.set_spacing_button.clicked.connect(self.set_peak_spacing)
        self.fit_basis_button.clicked.connect(self.fit_basis_vectors)
        self.accept_fit_button.clicked.connect(lambda: self.accept_stage("basis_fit"))
        self.accept_reference_button.clicked.connect(self.review_and_accept_reference)
        self.pick_roi_button.clicked.connect(self.start_roi_pick)
        self.export_button.clicked.connect(self.export_result)
        self.color_mode.currentTextChanged.connect(lambda _text: self._display_result())
        self.display_mode.currentTextChanged.connect(lambda _text: self._display_result())
        self.color_min_spin.valueChanged.connect(lambda _value: self._display_result())
        self.color_max_spin.valueChanged.connect(lambda _value: self._display_result())
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def notify_braggvectors_ready(self) -> None:
        self.status_label.setText("BraggVectors available")
        self._refresh_qr_rotation_summary()

    def run_strain_map(self) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            QMessageBox.information(self, "Strain Map", "Run full BraggVectors first.")
            return
        self._refresh_qr_rotation_summary()
        warning = self._calibration_warning(braggvectors)
        if warning:
            self.log_panel.log(f"WARN  {warning}")
            self.status_label.setText(warning)

        self.status_label.setText(f"Running... {warning}" if warning else "Running...")
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
        status = self.service.calibration_status(braggvectors)
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
        rotation_detail = ""
        if "rotation" in missing and status.rotate != "missing":
            rotation_detail = " QR rotation metadata exists, but rotation is not applied."
        return (
            "Calibration is incomplete; strain will continue, but accuracy may be lower. "
            f"Missing/applied-off corrections: {', '.join(missing)}.{rotation_detail}"
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
        form.addRow("applied QR rotation", self.applied_qr_rotation_label)
        form.addRow("max_peak_spacing", self.max_spacing_spin)
        form.addRow("minAbsoluteIntensity", self.min_abs_spin)
        form.addRow("minRelativeIntensity", self.min_rel_spin)
        form.addRow("minSpacing", self.min_spacing_spin)
        form.addRow("edgeBoundary", self.edge_spin)
        form.addRow("maxNumPeaks", self.max_peaks_spin)
        form.addRow("manual origin index (-1 auto)", self.index_origin)
        form.addRow("manual g1 index (-1 auto)", self.index_g1)
        form.addRow("manual g2 index (-1 auto)", self.index_g2)
        form.addRow("reference mode", self.reference_mode)
        form.addRow("display", self.display_mode)
        form.addRow("color range", self.color_mode)
        form.addRow("manual color min", self.color_min_spin)
        form.addRow("manual color max", self.color_max_spin)
        form.addRow("reference ROI rx start", self.roi_rx_start)
        form.addRow("reference ROI rx end", self.roi_rx_end)
        form.addRow("reference ROI ry start", self.roi_ry_start)
        form.addRow("reference ROI ry end", self.roi_ry_end)
        form.addRow("manual g1 x", self.manual_g1_x)
        form.addRow("manual g1 y", self.manual_g1_y)
        form.addRow("manual g2 x", self.manual_g2_x)
        form.addRow("manual g2 y", self.manual_g2_y)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(controls)
        left_layout.addWidget(self.pick_roi_button)
        left_layout.addWidget(self.choose_basis_button)
        left_layout.addWidget(self.accept_basis_button)
        left_layout.addWidget(self.set_spacing_button)
        left_layout.addWidget(self.fit_basis_button)
        left_layout.addWidget(self.accept_fit_button)
        left_layout.addWidget(self.accept_reference_button)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.export_button)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        self.controls_panel = left
        layout = QHBoxLayout(self)
        layout.addWidget(self.workspace)

    def _basis_params(self) -> BasisSelectionParams:
        optional_index = lambda value: None if value < 0 else int(value)
        return BasisSelectionParams(
            index_origin=optional_index(self.index_origin.value()),
            index_g1=optional_index(self.index_g1.value()),
            index_g2=optional_index(self.index_g2.value()),
            min_absolute_intensity=self.min_abs_spin.value(),
            min_relative_intensity=self.min_rel_spin.value(),
            min_spacing=self.min_spacing_spin.value(),
            edge_boundary=self.edge_spin.value(),
            max_num_peaks=self.max_peaks_spin.value(),
        )

    def choose_basis_vectors(self) -> None:
        try:
            result = self.service.choose_strain_basis(
                self.braggvectors_provider(), self._basis_params()
            )
            self._display_stage(result)
        except Exception as exc:
            self._handle_failed(str(exc))

    def set_peak_spacing(self) -> None:
        try:
            self._display_stage(self.service.set_strain_peak_spacing(self.max_spacing_spin.value()))
        except Exception as exc:
            self._handle_failed(str(exc))

    def fit_basis_vectors(self) -> None:
        try:
            self._display_stage(self.service.fit_strain_basis())
        except Exception as exc:
            self._handle_failed(str(exc))

    def _display_stage(self, result: StrainStageResult) -> None:
        quality = ", ".join(f"{key}={value}" for key, value in result.quality.items())
        self.status_label.setText(f"{result.message} {quality}".strip())
        self.log_panel.log(f"Strain stage {result.stage}: {result.message} {quality}".strip())
        self.workspace.set_results([
            FigureResult(
                name,
                image,
                vectors=result.vectors.get(name),
                circles=result.circles.get(name),
                diagnostic=quality,
                scaling="linear",
            )
            for name, image in result.images.items()
        ])

    def accept_stage(self, stage: str) -> None:
        try:
            state = self.service.accept_strain_stage(stage)
        except Exception as exc:
            self._handle_failed(str(exc))
            return
        if not {"basis_selection", "basis_fit", "reference"}.issubset(
            self.service.accepted_strain_stages
        ):
            QMessageBox.information(
                self,
                "Strain Map",
                "Choose, review, and explicitly accept the basis selection, basis fit, and reference first.",
            )
            return
        self.status_label.setText(
            "Accepted stages: "
            f"basis selection={state.basis_selection}, basis fit={state.basis_fit}, "
            f"reference={state.reference}."
        )
        self.run_button.setEnabled(state.reference)

    def review_and_accept_reference(self) -> None:
        try:
            result = self.service.preview_strain_reference(
                self.braggvectors_provider(), self._params()
            )
            self._display_stage(result)
            self.accept_stage("reference")
        except Exception as exc:
            self._handle_failed(str(exc))

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
            manual_g1_x=self.manual_g1_x.value(),
            manual_g1_y=self.manual_g1_y.value(),
            manual_g2_x=self.manual_g2_x.value(),
            manual_g2_y=self.manual_g2_y.value(),
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
            for name, image in result.process_images.items():
                self.result_registry.register(
                    name,
                    "strain process",
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
        process_images = getattr(self.result, "process_images", {})
        if self.display_mode.currentText() == "Process" and process_images:
            process_vectors = getattr(self.result, "process_vectors", {})
            self.workspace.set_results([
                FigureResult(
                    name,
                    image,
                    vectors=process_vectors.get(name),
                    scaling="linear",
                )
                for name, image in process_images.items()
                if image is not None
            ])
            return
        self.workspace.set_results([
            FigureResult(
                name,
                image,
                levels=self._levels_for(image),
                colormap="PRGn" if name == "theta" else "RdBu_r",
                scaling="linear",
            )
            for name, image in self.result.components.items()
            if image is not None
        ])

    def _refresh_qr_rotation_summary(self) -> None:
        braggvectors = self.braggvectors_provider()
        status = self.service.calibration_status(braggvectors)
        applied = bool(getattr(braggvectors, "calstate", {}).get("rotate", False)) if braggvectors else False
        suffix = "applied" if applied else "not applied"
        self.applied_qr_rotation_label.setText(f"{status.rotate} ({suffix})")

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
            self.manual_g1_x,
            self.manual_g1_y,
            self.manual_g2_x,
            self.manual_g2_y,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.STRAIN_MAP, "valueChanged")
        self.workflow_state.watch(
            self.reference_mode, WorkflowStep.STRAIN_MAP, "currentTextChanged"
        )

    def _refresh_stale_status(self) -> None:
        self._refresh_qr_rotation_summary()
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
            "index_origin": self.index_origin.value(),
            "index_g1": self.index_g1.value(),
            "index_g2": self.index_g2.value(),
            "reference_mode": params.reference_mode,
            "display_mode": self.display_mode.currentText(),
            "roi_rx_start": params.roi_rx_start,
            "roi_rx_end": params.roi_rx_end,
            "roi_ry_start": params.roi_ry_start,
            "roi_ry_end": params.roi_ry_end,
            "manual_g1_x": params.manual_g1_x,
            "manual_g1_y": params.manual_g1_y,
            "manual_g2_x": params.manual_g2_x,
            "manual_g2_y": params.manual_g2_y,
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
            "manual_g1_x": self.manual_g1_x,
            "manual_g1_y": self.manual_g1_y,
            "manual_g2_x": self.manual_g2_x,
            "manual_g2_y": self.manual_g2_y,
        }
        int_controls = {
            "edge_boundary": self.edge_spin,
            "max_num_peaks": self.max_peaks_spin,
            "index_origin": self.index_origin,
            "index_g1": self.index_g1,
            "index_g2": self.index_g2,
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
        if "display_mode" in params:
            self.display_mode.setCurrentText(str(params["display_mode"]))
        if "color_mode" in params:
            self.color_mode.setCurrentText(str(params["color_mode"]))
