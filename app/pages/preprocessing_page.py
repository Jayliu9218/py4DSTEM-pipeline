from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.services.preprocessing_service import HotPixelParams, HotPixelPreview, PreprocessingService
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.scientific_controls import (
    ScientificControlsPanel,
    action_row,
    property_row,
    section,
    status_row,
)
from app.widgets.worker_runner import WorkerRunner


class PreprocessingPage(QWidget, WorkerRunner):
    scan_overview_ready = Signal(object, object)

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        selected_source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.selected_source_provider = selected_source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = PreprocessingService()
        self._init_worker_runner()
        self.preview: HotPixelPreview | None = None
        self._show_data_source = None
        self.threshold = NumericLineEdit(1.01, 1000, 8, decimals=2, unit="x")
        self.preview_button = QPushButton("Preview Hot Pixels")
        self.apply_button = QPushButton("Apply Hot-Pixel Filter")
        self.apply_button.setEnabled(False)
        self.show_data_button = QPushButton("Show Data")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.memory_budget_mb = NumericLineEdit(8, 1024, 64, decimals=0, unit="MB", integer=True)
        self.preview_scan_stride = NumericLineEdit(1, 64, 1, decimals=0, integer=True)
        self.status = QLabel("Load and assign a Target DataCube.")
        self.status.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()
        self.preview_button.clicked.connect(self.preview_hot_pixels)
        self.apply_button.clicked.connect(self.apply_hot_pixels)
        self.show_data_button.clicked.connect(self.show_data)
        self.cancel_button.clicked.connect(self.cancel_background)
        self.threshold.valueChanged.connect(self._threshold_changed)
        self.controls_panel = ScientificControlsPanel([
            section("Data Preprocessing", [
                property_row("hot-pixel threshold", self.threshold),
                property_row("reduction memory budget", self.memory_budget_mb),
                property_row("preview scan stride", self.preview_scan_stride),
                property_row("", action_row(self.show_data_button, self.cancel_button)),
                property_row("", action_row(self.preview_button, self.apply_button)),
            ], number=1),
            section("Status", [
                property_row("", status_row(self.status)),
            ]),
        ])
        layout = QVBoxLayout(self)
        layout.addWidget(self.workspace)

    def show_data(self) -> None:
        source = self.selected_source_provider()
        if source is None:
            source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Preprocess", "Select or assign a displayable data object first.")
            return
        budget_mb = int(self.memory_budget_mb.value())
        scan_stride = int(self.preview_scan_stride.value())
        self._show_data_source = source
        task = self.service.show_data_task(source, memory_budget_mb=budget_mb, scan_stride=scan_stride)
        self._start_background(
            task.name,
            task,
            capture_stdout=False,
        )

    def _on_start(self, name: str) -> None:
        super()._on_start(name)
        self.show_data_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

    def _handle_result(self, result) -> None:
        if self.pending_operation == "Show Data":
            self.workspace.append_results([
                FigureResult(name, image) for name, image in result.items()
            ])
            self.status.setText("Data display ready.")
            scan_overview = result.get("Scan overview")
            if scan_overview is not None:
                self.scan_overview_ready.emit(self._show_data_source, scan_overview)
            self.log_panel.process_finished(self.pending_operation)
        else:
            super()._handle_result(result)
        self.show_data_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._show_data_source = None

    def _handle_error(self, message: str) -> None:
        super()._handle_error(message)
        self.show_data_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._show_data_source = None

    def _handle_cancelled(self) -> None:
        super()._handle_cancelled()
        self.show_data_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._show_data_source = None

    def preview_hot_pixels(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Preprocess", "Load a Target DataCube first.")
            return
        try:
            self.preview = self.service.preview_hot_pixels(
                source, HotPixelParams(self.threshold.value())
            )
        except Exception as exc:
            QMessageBox.warning(self, "Preprocess", str(exc))
            return
        self.workspace.append_results([
            FigureResult("Mean diffraction before", self.preview.before_mean),
            FigureResult("Detected hot pixels", self.preview.hot_pixel_mask.astype(float)),
            FigureResult("Mean diffraction preview after", self.preview.after_mean),
        ])
        self.apply_button.setEnabled(True)
        self.status.setText(f"Detected {self.preview.hot_pixel_count} hot pixels. Review, then Apply.")

    def apply_hot_pixels(self) -> None:
        source = self.source_provider()
        if source is None or self.preview is None:
            return
        try:
            count = self.service.apply_hot_pixels(source, self.preview)
        except Exception as exc:
            QMessageBox.warning(self, "Preprocess", str(exc))
            return
        self.apply_button.setEnabled(False)
        self.workflow_state.mark_completed(WorkflowStep.PREPROCESS_APPLY)
        self.log_panel.log(f"Applied hot-pixel preprocessing: {count} pixels replaced.")
        self.status.setText(f"Applied hot-pixel preprocessing to {count} pixels.")
        if self.result_registry is not None:
            self.result_registry.register(
                "hot pixel mask", "Preprocessing", self.preview.hot_pixel_mask.astype(float),
                ("npy", "png", "tiff"), self.params_snapshot(),
            )

    def _threshold_changed(self, _value: float) -> None:
        self.preview = None
        self.apply_button.setEnabled(False)
        self.workflow_state.parameters_updated(WorkflowStep.PREPROCESS_APPLY)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "hot_pixel_threshold": self.threshold.value(),
            "reduction_memory_budget_mb": int(self.memory_budget_mb.value()),
            "preview_scan_stride": int(self.preview_scan_stride.value()),
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        if "hot_pixel_threshold" in params:
            self.threshold.setValue(float(params["hot_pixel_threshold"]))
        if "reduction_memory_budget_mb" in params:
            self.memory_budget_mb.setValue(int(params["reduction_memory_budget_mb"]))
        if "preview_scan_stride" in params:
            self.preview_scan_stride.setValue(int(params["preview_scan_stride"]))
