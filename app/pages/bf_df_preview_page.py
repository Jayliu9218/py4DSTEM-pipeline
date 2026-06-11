from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.phase_contrast_service import PhaseContrastService, PhaseContrastServiceError
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.progress_stream import ProgressStream


class BFDFWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            stream = ProgressStream(self.progress.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                self.finished.emit(self.operation())
            stream.flush()
        except Exception as exc:
            self.failed.emit(str(exc))


class BFDFPreviewPage(QWidget):
    bf_df_ready = Signal(dict)

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
        self.service = PhaseContrastService()
        self.worker_thread: QThread | None = None
        self.worker: BFDFWorker | None = None
        self.result: dict[str, np.ndarray] | None = None

        self.bf_radius = self._float_input(1, 1000, 10, decimals=1, unit="px")
        self.df_inner = self._float_input(1, 1000, 20, decimals=1, unit="px")
        self.df_outer = self._float_input(1, 1000, 50, decimals=1, unit="px")
        self.contrast_low = self._float_input(0, 100, 1, decimals=1, unit="%")
        self.contrast_high = self._float_input(0, 100, 99, decimals=1, unit="%")
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.compute_button = QPushButton("Compute BF / DF")
        self.compute_button.clicked.connect(self._run)

        self.workspace = AdaptiveImageWorkspace()

        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def _build_layout(self) -> None:
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("BF Radius", self.bf_radius)
        form.addRow("DF Inner Radius", self.df_inner)
        form.addRow("DF Outer Radius", self.df_outer)
        form.addRow("Contrast Low %", self.contrast_low)
        form.addRow("Contrast High %", self.contrast_high)
        layout.addLayout(form)
        layout.addWidget(self.compute_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.controls_panel = controls

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.workspace, 1)

    def _run(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "BF / DF Preview", "Load a 4D DataCube first.")
            return

        self.status_label.setText("Computing BF / DF...")
        self.compute_button.setEnabled(False)
        self.log_panel.log("BF / DF preview started")

        bf_radius = self.bf_radius.value()
        df_inner = self.df_inner.value()
        df_outer = self.df_outer.value()
        probe_geom = self.probe_geometry_provider()

        operation = lambda: self.service.compute_bf_df(
            source,
            bf_radius=bf_radius,
            df_inner=df_inner,
            df_outer=df_outer,
            probe_geometry=probe_geom,
        )

        self.worker_thread = QThread()
        self.worker = BFDFWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.progress.connect(lambda msg: self.log_panel.log(msg))
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _handle_finished(self, images: dict[str, np.ndarray]) -> None:
        self.result = images
        self.compute_button.setEnabled(True)
        self.status_label.setText("Done")
        self.log_panel.log("BF / DF preview complete")

        for name, image in images.items():
            self.workspace.append_result(FigureResult(f"BF / DF: {name}", image))

        if self.result_registry is not None:
            for name, image in images.items():
                key = name.lower().replace(" ", "_")
                self.result_registry.register(
                    name=f"phase_retrieval_{key}",
                    category="Phase Retrieval",
                    data=image,
                    export_formats=("npy", "png", "tiff"),
                )

        self.bf_df_ready.emit(images)
        self.workflow_state.mark_completed(WorkflowStep.BF_DF_PREVIEW)

    def _handle_failed(self, error: str) -> None:
        self.compute_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"BF / DF preview failed: {error}")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.BF_DF_PREVIEW]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "bf_radius": self.bf_radius.value(),
            "df_inner": self.df_inner.value(),
            "df_outer": self.df_outer.value(),
            "contrast_low": self.contrast_low.value(),
            "contrast_high": self.contrast_high.value(),
        }
