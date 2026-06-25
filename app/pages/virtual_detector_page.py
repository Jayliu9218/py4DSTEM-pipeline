from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.virtual_detector_service import (
    VirtualDetectorParams,
    VirtualDetectorResult,
    VirtualDetectorService,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class VirtualDetectorPage(QWidget, WorkerRunner):
    virtual_image_ready = Signal(object)

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        shape_provider: Callable[[], tuple[int, int, int, int] | None],
        probe_geometry_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.shape_provider = shape_provider
        self.probe_geometry_provider = probe_geometry_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = VirtualDetectorService()
        self.result: np.ndarray | None = None
        self._init_worker_runner()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                VirtualDetectorService.BRIGHT_FIELD,
                VirtualDetectorService.ANNULAR_DARK_FIELD,
                VirtualDetectorService.CUSTOM_ANNULAR,
                VirtualDetectorService.OFF_AXIS_DARK_FIELD,
                VirtualDetectorService.VIRTUAL_DIFFRACTION,
            ]
        )

        self.center_x_spin = self._make_float_input(0, 100000, 1, unit="px")
        self.center_y_spin = self._make_float_input(0, 100000, 1, unit="px")
        self.inner_radius_spin = self._make_float_input(0, 100000, 0, unit="px")
        self.outer_radius_spin = self._make_float_input(0.1, 100000, 10, unit="px")
        self.roi_rx_start = self._make_float_input(0, 100000, 0, unit="px")
        self.roi_rx_end = self._make_float_input(1, 100000, 1, unit="px")
        self.roi_ry_start = self._make_float_input(0, 100000, 0, unit="px")
        self.roi_ry_end = self._make_float_input(1, 100000, 1, unit="px")
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["rectangle", "circle"])
        self.roi_center_x = self._make_float_input(0, 100000, 84, unit="px")
        self.roi_center_y = self._make_float_input(0, 100000, 30, unit="px")
        self.roi_radius = self._make_float_input(0.1, 100000, 10, unit="px")
        self.notebook_preset_button = QPushButton("Apply 01_CBS Geometry")

        self.run_button = QPushButton("Plot")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()

        self.run_button.clicked.connect(self.run_detector)
        self.notebook_preset_button.clicked.connect(self.apply_notebook_geometry)
        self.export_button.clicked.connect(self.export_result)
        self.mode_combo.currentTextChanged.connect(self._sync_mode_state)
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)

        self._build_layout()
        self._sync_mode_state()

    def _build_layout(self) -> None:
        controls = QWidget()
        form = QFormLayout(controls)
        self._controls_form = form
        form.addRow("Mode", self.mode_combo)
        form.addRow("center_x", self.center_x_spin)
        form.addRow("center_y", self.center_y_spin)
        form.addRow("inner_radius", self.inner_radius_spin)
        form.addRow("outer_radius", self.outer_radius_spin)
        form.addRow("", self.notebook_preset_button)
        form.addRow("real-space ROI mode", self.roi_mode_combo)
        form.addRow("ROI rx start", self.roi_rx_start)
        form.addRow("ROI rx end", self.roi_rx_end)
        form.addRow("ROI ry start", self.roi_ry_start)
        form.addRow("ROI ry end", self.roi_ry_end)
        form.addRow("ROI center x", self.roi_center_x)
        form.addRow("ROI center y", self.roi_center_y)
        form.addRow("ROI radius", self.roi_radius)

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.export_button)

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(controls)
        self.left_layout.addLayout(button_row)
        self.left_layout.addWidget(self.status_label)
        self.left_layout.addStretch(1)

        left = QGroupBox("Virtual Detector")
        left.setLayout(self.left_layout)
        self.controls_panel = left

        layout = QHBoxLayout(self)
        layout.addWidget(self.workspace)

    def add_controls_widget(self, widget: QWidget) -> None:
        self.left_layout.insertWidget(max(self.left_layout.count() - 1, 0), widget)

    def _make_float_input(
        self,
        minimum: float,
        maximum: float,
        value: float,
        unit: str = "",
    ) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=2, unit=unit)

    def refresh_defaults_from_datacube(self) -> None:
        shape = self.shape_provider()
        if shape is None or len(shape) != 4:
            return

        qx, qy = shape[2], shape[3]
        geometry = self.probe_geometry_provider()
        if geometry is None:
            self.center_x_spin.setValue((qx - 1) / 2)
            self.center_y_spin.setValue((qy - 1) / 2)
            self.inner_radius_spin.setValue(max(min(qx, qy) * 0.08, 1))
            self.outer_radius_spin.setValue(max(min(qx, qy) * 0.25, 2))
            self.log_panel.log("Virtual detector defaults updated from diffraction shape.")
            return

        self.center_x_spin.setValue(geometry.center_x)
        self.center_y_spin.setValue(geometry.center_y)
        self.inner_radius_spin.setValue(geometry.radius * 1.5)
        self.outer_radius_spin.setValue(geometry.radius * 3.0)
        self.log_panel.log(
            "Virtual detector defaults updated from measured probe center and radius."
        )

    def apply_notebook_geometry(self) -> None:
        geometry = self.probe_geometry_provider()
        if geometry is None:
            self.refresh_defaults_from_datacube()
            self.status_label.setText("Measure probe geometry before applying the 01_CBS detector preset.")
            return
        mode = self.mode_combo.currentText()
        self.center_x_spin.setValue(geometry.center_x)
        self.center_y_spin.setValue(geometry.center_y)
        if mode == VirtualDetectorService.ANNULAR_DARK_FIELD:
            self.inner_radius_spin.setValue(3 * geometry.radius)
            self.outer_radius_spin.setValue(6 * geometry.radius)
        elif mode == VirtualDetectorService.OFF_AXIS_DARK_FIELD:
            self.center_y_spin.setValue(geometry.center_y / 3)
            self.outer_radius_spin.setValue(1.25 * geometry.radius)
        else:
            self.inner_radius_spin.setValue(0)
            self.outer_radius_spin.setValue(geometry.radius)
        self.status_label.setText(f"Applied editable 01_CBS geometry for {mode}.")

    def run_detector(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Virtual Detector", "Load a 4D DataCube first.")
            return

        params = self._params_from_ui()
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.log_panel.log(f"Virtual detector started: {params.mode}")

        task = self.service.compute_task(source, params)
        if not self._start_background(
            task.name,
            task,
            capture_stdout=False,
        ):
            self.run_button.setEnabled(True)

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
            roi_rx_start=int(self.roi_rx_start.value()),
            roi_rx_end=int(self.roi_rx_end.value()),
            roi_ry_start=int(self.roi_ry_start.value()),
            roi_ry_end=int(self.roi_ry_end.value()),
            roi_mode=self.roi_mode_combo.currentText(),
            roi_center_x=self.roi_center_x.value(),
            roi_center_y=self.roi_center_y.value(),
            roi_radius=self.roi_radius.value(),
        )

    def _handle_result(self, result: VirtualDetectorResult) -> None:
        self.result = result.image
        figures = []
        if result.real_space_preview is not None:
            figures.append(FigureResult(
                "Real-space ROI",
                result.real_space_preview,
                overlay=result.real_space_overlay,
            ))
        if result.detector_preview is not None:
            figures.append(
                FigureResult(
                    "Detector Position",
                    result.detector_preview,
                    overlay=result.detector_overlay,
                )
            )
        figures.append(FigureResult(result.mode, result.image))
        self.workspace.append_results(figures)
        if result.mode != VirtualDetectorService.VIRTUAL_DIFFRACTION:
            self.virtual_image_ready.emit(result.image)
        self.status_label.setText(f"Done in {result.elapsed_seconds:.2f} s")
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.log_panel.log(
            f"Virtual detector completed: {result.mode}, elapsed={result.elapsed_seconds:.2f} s"
        )
        self.log_panel.process_finished(
            "Virtual detector", f"{result.mode}, elapsed={result.elapsed_seconds:.2f} s"
        )
        self.workflow_state.mark_completed(WorkflowStep.VIRTUAL_DETECTOR)
        if self.result_registry is not None:
            self.result_registry.register(
                "virtual detector image",
                "Check data",
                result.image,
                ("npy", "png", "tiff"),
                {"mode": result.mode, **self.params_snapshot()},
            )
            if result.detector_preview is not None:
                self.result_registry.register(
                    "virtual detector position",
                    "Check data",
                    result.detector_preview,
                    ("npy", "png", "tiff"),
                    {"mode": result.mode, "overlay": result.detector_overlay, **self.params_snapshot()},
                )

    def _handle_error(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(self.result is not None)
        self.log_panel.log(f"Virtual detector failed: {message}")
        self.log_panel.process_failed(self.pending_operation or "Virtual detector", message)
        QMessageBox.warning(self, "Virtual Detector", message)

    def _sync_mode_state(self) -> None:
        mode = self.mode_combo.currentText()
        is_circle = mode in {VirtualDetectorService.BRIGHT_FIELD, VirtualDetectorService.OFF_AXIS_DARK_FIELD}
        is_diffraction = mode == VirtualDetectorService.VIRTUAL_DIFFRACTION
        self.inner_radius_spin.setEnabled(not is_circle and not is_diffraction)
        for control in [self.center_x_spin, self.center_y_spin, self.inner_radius_spin, self.outer_radius_spin]:
            self._set_form_control_visible(control, not is_diffraction)
        for control in [
            self.roi_rx_start, self.roi_rx_end, self.roi_ry_start, self.roi_ry_end,
            self.roi_mode_combo, self.roi_center_x, self.roi_center_y, self.roi_radius,
        ]:
            self._set_form_control_visible(control, is_diffraction)

    def _set_form_control_visible(self, control: QWidget, visible: bool) -> None:
        control.setVisible(visible)
        label = self._controls_form.labelForField(control)
        if label is not None:
            label.setVisible(visible)

    def _watch_parameters(self) -> None:
        self.workflow_state.watch(
            self.mode_combo, WorkflowStep.VIRTUAL_DETECTOR, "currentTextChanged"
        )
        for spin in [
            self.center_x_spin,
            self.center_y_spin,
            self.inner_radius_spin,
            self.outer_radius_spin,
            self.roi_rx_start,
            self.roi_rx_end,
            self.roi_ry_start,
            self.roi_ry_end,
            self.roi_center_x,
            self.roi_center_y,
            self.roi_radius,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.VIRTUAL_DETECTOR, "valueChanged")
        self.workflow_state.watch(self.roi_mode_combo, WorkflowStep.VIRTUAL_DETECTOR, "currentTextChanged")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.is_stale(WorkflowStep.VIRTUAL_DETECTOR):
            self.status_label.setText(STALE_RESULTS_MESSAGE)

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
            try:
                import tifffile
            except ModuleNotFoundError as exc:
                raise ValueError(
                    "TIFF export requires tifffile. Install project requirements first."
                ) from exc
            tifffile.imwrite(path, np.asarray(self.result))
        elif path.suffix.lower() == ".png":
            import matplotlib.pyplot as plt

            plt.imsave(path, np.asarray(self.result), cmap="gray")
        else:
            raise ValueError("Supported export formats are PNG, TIFF, and NPY.")

    def params_snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode_combo.currentText(),
            "center_x": self.center_x_spin.value(),
            "center_y": self.center_y_spin.value(),
            "inner_radius": self.inner_radius_spin.value(),
            "outer_radius": self.outer_radius_spin.value(),
            "roi_rx_start": int(self.roi_rx_start.value()),
            "roi_rx_end": int(self.roi_rx_end.value()),
            "roi_ry_start": int(self.roi_ry_start.value()),
            "roi_ry_end": int(self.roi_ry_end.value()),
            "roi_mode": self.roi_mode_combo.currentText(),
            "roi_center_x": self.roi_center_x.value(),
            "roi_center_y": self.roi_center_y.value(),
            "roi_radius": self.roi_radius.value(),
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        self.mode_combo.setCurrentText(str(params.get("mode", self.mode_combo.currentText())))
        for key, spin in [
            ("center_x", self.center_x_spin),
            ("center_y", self.center_y_spin),
            ("inner_radius", self.inner_radius_spin),
            ("outer_radius", self.outer_radius_spin),
            ("roi_rx_start", self.roi_rx_start),
            ("roi_rx_end", self.roi_rx_end),
            ("roi_ry_start", self.roi_ry_start),
            ("roi_ry_end", self.roi_ry_end),
            ("roi_center_x", self.roi_center_x),
            ("roi_center_y", self.roi_center_y),
            ("roi_radius", self.roi_radius),
        ]:
            if key in params:
                spin.setValue(float(params[key]))
        if "roi_mode" in params:
            self.roi_mode_combo.setCurrentText(str(params["roi_mode"]))
