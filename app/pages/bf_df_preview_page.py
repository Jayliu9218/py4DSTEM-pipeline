from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
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
from app.services.result_cache import ResultCache, make_result_cache_key
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.theme import Theme
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class BFDFPreviewPage(QWidget, WorkerRunner):
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
        self.result_cache = ResultCache(limit=4)
        self._init_worker_runner()
        self.result: dict[str, np.ndarray] | None = None
        self._pending_cache_key = None

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

        self.compute_button.setEnabled(False)
        self.log_panel.log("BF / DF preview started")

        bf_radius = self.bf_radius.value()
        df_inner = self.df_inner.value()
        df_outer = self.df_outer.value()
        probe_geom = self.probe_geometry_provider()
        cache_key = self._cache_key(source)
        cached = self.result_cache.get(cache_key)
        if cached is not None:
            self.log_panel.log("BF / DF preview reused cached result")
            self._display_result(cached, from_cache=True)
            return

        self._pending_cache_key = cache_key
        task = self.service.compute_bf_df_task(
            source,
            bf_radius=bf_radius,
            df_inner=df_inner,
            df_outer=df_outer,
            probe_geometry=probe_geom,
        )
        if not self._start_background(task.name, task):
            self.compute_button.setEnabled(True)
            self._pending_cache_key = None

    def _handle_result(self, images: dict[str, np.ndarray]) -> None:
        if self._pending_cache_key is not None:
            self.result_cache.put(self._pending_cache_key, images)
        self._pending_cache_key = None
        self._display_result(images)

    def _display_result(self, images: dict[str, np.ndarray], *, from_cache: bool = False) -> None:
        self.result = images
        self.compute_button.setEnabled(True)
        self.status_label.setText("Done" if not from_cache else "Done (cached)")
        if not from_cache:
            self.log_panel.log("BF / DF preview complete")
            self.log_panel.process_finished(self.pending_operation)

        self.workspace.append_results([
            FigureResult(f"BF / DF: {name}", image) for name, image in images.items()
        ])

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

    def _handle_error(self, error: str) -> None:
        self._pending_cache_key = None
        self.compute_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"BF / DF preview failed: {error}")
        self.log_panel.process_failed(self.pending_operation or "BF / DF Preview", error)

    def _handle_progress(self, message: str, fraction: float) -> None:
        # BF/DF forwards captured text to both the activity log and the progress bar.
        self.log_panel.log(message)
        super()._handle_progress(message, fraction)

    def _cache_key(self, source) -> object:
        return make_result_cache_key("bf_df_preview", source, self.params_snapshot())

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.BF_DF_PREVIEW]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet(f"color: {Theme.STALE};")

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
