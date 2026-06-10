from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit


class BraggVectorMapPage(QWidget):
    braggvectors_viewed = Signal = None

    def __init__(
        self,
        braggvectors_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.braggvectors_provider = braggvectors_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry

        self.status_label = QLabel("Idle — run Bragg Detection first")
        self.status_label.setWordWrap(True)

        self.colormap = QComboBox()
        self.colormap.addItems(["gray", "viridis", "magma", "plasma", "inferno", "cividis"])
        self.percentile_low = NumericLineEdit(0, 100, 1, decimals=1, unit="%")
        self.percentile_high = NumericLineEdit(0, 100, 99, decimals=1, unit="%")

        self.bvm_button = QPushButton("Generate Bragg Vector Map")
        self.bvm_button.clicked.connect(self._generate_bvm)
        self.peak_count_button = QPushButton("Generate Peak Count Map")
        self.peak_count_button.clicked.connect(self._generate_peak_count)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(0, 0)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()

    def _build_layout(self) -> None:
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("Colormap", self.colormap)
        form.addRow("Percentile Low", self.percentile_low)
        form.addRow("Percentile High", self.percentile_high)
        layout.addLayout(form)
        layout.addWidget(self.bvm_button)
        layout.addWidget(self.peak_count_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.controls_panel = controls

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.viewer, 1)

    def _generate_bvm(self) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            self.status_label.setText("No BraggVectors available. Run Bragg Detection first.")
            return

        try:
            from py4DSTEM.process.utils import get_bragg_vector_map
            Rx, Ry = braggvectors.Rshape if hasattr(braggvectors, 'Rshape') else braggvectors.shape[:2]
            Qx, Qy = braggvectors.Qshape if hasattr(braggvectors, 'Qshape') else braggvectors.shape[2:]
        except ImportError:
            try:
                pointlistarray = braggvectors.raw._data if hasattr(braggvectors, 'raw') else braggvectors._data
                Rx = pointlistarray.shape[0]
                Ry = pointlistarray.shape[1]
                Qx = max(pointlistarray.max('qx'), 1) if hasattr(pointlistarray, 'max') else 256
                Qy = max(pointlistarray.max('qy'), 1) if hasattr(pointlistarray, 'max') else 256
            except Exception:
                self.status_label.setText("Could not determine BraggVector map dimensions.")
                return

        try:
            import numpy as np
            bvm = np.zeros((Rx, Ry), dtype=float)
            raw = braggvectors.raw if hasattr(braggvectors, 'raw') else braggvectors
            data = raw._data if hasattr(raw, '_data') else raw
            if hasattr(braggvectors, 'Rshape'):
                for rx in range(Rx):
                    for ry in range(Ry):
                        try:
                            pl = data[rx, ry]
                            if hasattr(pl, 'length'):
                                bvm[rx, ry] = pl.length
                            elif hasattr(pl, '__len__'):
                                bvm[rx, ry] = len(pl)
                        except Exception:
                            pass
            p_low = self.percentile_low.value()
            p_high = self.percentile_high.value()
            vmax = np.percentile(bvm[bvm > 0], p_high) if np.any(bvm > 0) else 1
            vmin = np.percentile(bvm[bvm > 0], p_low) if np.any(bvm > 0) else 0
            self.viewer.set_image(bvm)
            self.status_label.setText(f"BVM generated: shape={bvm.shape}, range=[{vmin:.1f}, {vmax:.1f}]")
            self.log_panel.log(f"Bragg Vector Map generated: shape={bvm.shape}")

            if self.result_registry is not None:
                self.result_registry.register(
                    name="bragg_vector_map",
                    category="Bragg Detection",
                    data=bvm,
                    export_formats=("npy", "png", "tiff"),
                )

            self.workflow_state.mark_completed(WorkflowStep.BRAGG_VECTOR_MAP)
        except Exception as exc:
            self.status_label.setText(f"BVM generation failed: {exc}")
            self.log_panel.log(f"Bragg Vector Map generation failed: {exc}")

    def _generate_peak_count(self) -> None:
        braggvectors = self.braggvectors_provider()
        if braggvectors is None:
            self.status_label.setText("No BraggVectors available. Run Bragg Detection first.")
            return

        try:
            import numpy as np
            Rx, Ry = braggvectors.Rshape if hasattr(braggvectors, 'Rshape') else braggvectors.shape[:2]
            peak_count = np.zeros((Rx, Ry), dtype=float)
            data = braggvectors.raw._data if hasattr(braggvectors, 'raw') else braggvectors._data
            for rx in range(Rx):
                for ry in range(Ry):
                    try:
                        pl = data[rx, ry]
                        if hasattr(pl, 'length'):
                            peak_count[rx, ry] = pl.length
                        elif hasattr(pl, '__len__'):
                            peak_count[rx, ry] = len(pl)
                    except Exception:
                        pass
            self.viewer.set_image(peak_count)
            self.status_label.setText(f"Peak Count Map generated: shape={peak_count.shape}")
            self.log_panel.log(f"Peak Count Map generated: shape={peak_count.shape}")

            if self.result_registry is not None:
                self.result_registry.register(
                    name="peak_count_map",
                    category="Bragg Detection",
                    data=peak_count,
                    export_formats=("npy", "png", "tiff"),
                )
        except Exception as exc:
            self.status_label.setText(f"Peak count generation failed: {exc}")
            self.log_panel.log(f"Peak Count Map generation failed: {exc}")

    def _refresh_stale_status(self) -> None:
        if self.workflow_state.any_stale([WorkflowStep.BRAGG_VECTOR_MAP]):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def params_snapshot(self) -> dict[str, object]:
        return {
            "bragg_vector_map_colormap": self.colormap.currentText(),
            "bragg_vector_map_percentile_low": self.percentile_low.value(),
            "bragg_vector_map_percentile_high": self.percentile_high.value(),
        }