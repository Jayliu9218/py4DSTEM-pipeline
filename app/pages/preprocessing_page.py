from __future__ import annotations

from typing import Callable

import PySide6
from PySide6.QtWidgets import QFormLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.services.preprocessing_service import HotPixelParams, HotPixelPreview, PreprocessingService
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit


class PreprocessingPage(QWidget):
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
        self.preview: HotPixelPreview | None = None
        self.threshold = NumericLineEdit(1.01, 1000, 8, decimals=2, unit="x")
        self.preview_button = QPushButton("Preview Hot Pixels")
        self.apply_button = QPushButton("Apply Hot-Pixel Filter")
        self.apply_button.setEnabled(False)
        self.diagnostics_button = QPushButton("Show Datacube")
        self.show_selected_button = QPushButton("Show Selected Data")
        self.status = QLabel("Load and assign a Target DataCube.")
        self.status.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()
        self.preview_button.clicked.connect(self.preview_hot_pixels)
        self.apply_button.clicked.connect(self.apply_hot_pixels)
        self.diagnostics_button.clicked.connect(self.show_diagnostics)
        self.show_selected_button.clicked.connect(self.show_selected_data)
        self.threshold.valueChanged.connect(self._threshold_changed)
        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow("hot-pixel threshold", self.threshold)
        control_layout = QVBoxLayout()
        control_layout.addWidget(controls)
        
        """      
        control_layout.addWidget(self.diagnostics_button)
        control_layout.addWidget(self.show_selected_button)
        """
        button_row = PySide6.QtWidgets.QHBoxLayout()
        button_row.addWidget(self.diagnostics_button)
        button_row.addWidget(self.show_selected_button)
        control_layout.addLayout(button_row)
        
        """
        control_layout.addWidget(self.preview_button)
        control_layout.addWidget(self.apply_button)
        """
        
        button_row = PySide6.QtWidgets.QHBoxLayout()
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.apply_button)
        control_layout.addLayout(button_row)
        
        control_layout.addWidget(self.status)
        control_layout.addStretch(1)
        self.controls_panel = QWidget()
        self.controls_panel.setLayout(control_layout)
        layout = QVBoxLayout(self)
        layout.addWidget(self.workspace)

    def show_selected_data(self) -> None:
        source = self.selected_source_provider()
        if source is None:
            source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Preprocess", "Select or assign a displayable data object first.")
            return
        try:
            images = self.service.display_data(source)
        except Exception as exc:
            QMessageBox.warning(self, "Preprocess", str(exc))
            return
        self.workspace.append_results([FigureResult(name, image) for name, image in images.items()])
        self.status.setText("Selected data display ready.")

    def show_diagnostics(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Preprocess", "Load a Target DataCube first.")
            return
        try:
            images = self.service.basic_diagnostics(source)
        except Exception as exc:
            QMessageBox.warning(self, "Preprocess", str(exc))
            return
        self.workspace.append_results([FigureResult(name, image) for name, image in images.items()])
        self.status.setText("Basic DataCube diagnostics ready.")

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
        return {"hot_pixel_threshold": self.threshold.value()}

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        if "hot_pixel_threshold" in params:
            self.threshold.setValue(float(params["hot_pixel_threshold"]))
