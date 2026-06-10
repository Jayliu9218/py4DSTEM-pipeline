from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.phase_contrast_service import PhaseContrastResult
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class MethodComparisonPage(QWidget):
    def __init__(
        self,
        dpc_result_provider: Callable[[], PhaseContrastResult | None],
        parallax_result_provider: Callable[[], PhaseContrastResult | None],
        ptychography_result_provider: Callable[[], PhaseContrastResult | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
    ) -> None:
        super().__init__()
        self.dpc_result_provider = dpc_result_provider
        self.parallax_result_provider = parallax_result_provider
        self.ptychography_result_provider = ptychography_result_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state

        self.dpc_viewer = ImageViewer()
        self.dpc_viewer.setMinimumSize(0, 0)
        self.dpc_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.parallax_viewer = ImageViewer()
        self.parallax_viewer.setMinimumSize(0, 0)
        self.parallax_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ptychography_viewer = ImageViewer()
        self.ptychography_viewer.setMinimumSize(0, 0)
        self.ptychography_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.refresh_button = QPushButton("Refresh Results")
        self.refresh_button.clicked.connect(self._refresh)

        self.status_label = QLabel("Run DPC, Parallax, or Ptychography first.")
        self.status_label.setWordWrap(True)

        self._build_layout()
        self.workflow_state.changed.connect(self._update_status)

    def _build_layout(self) -> None:
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.controls_panel = controls

        dpc_label = QLabel("DPC / CoM")
        dpc_label.setObjectName("viewerTitle")
        parallax_label = QLabel("Parallax")
        parallax_label.setObjectName("viewerTitle")
        ptychography_label = QLabel("Ptychography")
        ptychography_label.setObjectName("viewerTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(dpc_label)
        layout.addWidget(self.dpc_viewer, 1)
        layout.addWidget(parallax_label)
        layout.addWidget(self.parallax_viewer, 1)
        layout.addWidget(ptychography_label)
        layout.addWidget(self.ptychography_viewer, 1)

    def _refresh(self) -> None:
        dpc = self.dpc_result_provider()
        if dpc is not None:
            image = dpc.images.get("Complex CoM")
            if image is None:
                image = dpc.images.get("Phase")
            if image is not None:
                self.dpc_viewer.set_image(np.asarray(image))
            else:
                first_image = next(iter(dpc.images.values()), None)
                if first_image is not None:
                    self.dpc_viewer.set_image(np.asarray(first_image))
        else:
            self.dpc_viewer.clear("No DPC result")

        parallax = self.parallax_result_provider()
        if parallax is not None:
            for name, image in parallax.images.items():
                self.parallax_viewer.set_image(np.asarray(image))
                break
        else:
            self.parallax_viewer.clear("No Parallax result")

        ptychography = self.ptychography_result_provider()
        if ptychography is not None:
            for name, image in ptychography.images.items():
                self.ptychography_viewer.set_image(np.asarray(image))
                break
        else:
            self.ptychography_viewer.clear("No Ptychography result")

        has_any = dpc is not None or parallax is not None or ptychography is not None
        if has_any:
            self.workflow_state.mark_completed(WorkflowStep.METHOD_COMPARISON)

    def _update_status(self) -> None:
        dpc_done = self.workflow_state.is_completed(WorkflowStep.DPC)
        parallax_done = self.workflow_state.is_completed(WorkflowStep.PARALLAX)
        ptychography_done = self.workflow_state.is_completed(WorkflowStep.PTYCHOGRAPHY)
        parts = []
        if dpc_done:
            parts.append("DPC: done")
        else:
            parts.append("DPC: not run")
        if parallax_done:
            parts.append("Parallax: done")
        else:
            parts.append("Parallax: not run")
        if ptychography_done:
            parts.append("Ptychography: done")
        else:
            parts.append("Ptychography: not run")
        self.status_label.setText("\n".join(parts))

    def params_snapshot(self) -> dict[str, object]:
        return {"method_comparison": True}