from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget

from app.services.result_registry import ResultEntry, ResultRegistry
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult


class CrystallineResultsPage(QWidget):
    FINAL_NAMES = {
        "orientation": {"orientation rgb"},
        "strain": {"strain map components", "exx", "eyy", "exy", "theta"},
    }
    QUALITY_WORDS = ("correlation", "confidence", "reliability", "peak count", "mask", "residual")

    def __init__(
        self,
        result_registry: ResultRegistry,
        orientation_mapping_controls: QWidget,
        orientation_mapping_refresh: Callable[[], None],
        workspace: AdaptiveImageWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.result_registry = result_registry
        self.orientation_mapping_refresh = orientation_mapping_refresh
        self.workspace = workspace or AdaptiveImageWorkspace()
        self.family = QComboBox()
        self.family.addItems(["Orientation", "Strain", "All"])
        self.view = QComboBox()
        self.view.addItems(["Final Results", "Quality Maps", "Process / Diagnostics"])
        self.refresh_button = QPushButton("Refresh Registered Results")
        self.refresh_button.clicked.connect(self.refresh_results)
        self.family.currentTextChanged.connect(self.refresh_results)
        self.view.currentTextChanged.connect(self.refresh_results)
        filters = QGroupBox("Results & Quality")
        form = QFormLayout(filters)
        form.addRow("Result family", self.family)
        form.addRow("View", self.view)
        form.addRow(self.refresh_button)
        self.mapping_group = QGroupBox("Orientation Mapping")
        mapping_layout = QVBoxLayout(self.mapping_group)
        mapping_layout.addWidget(orientation_mapping_controls)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(filters)
        controls_layout.addWidget(self.mapping_group)
        controls_layout.addStretch(1)
        self.controls_panel = controls
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.workspace)

    def refresh_stage(self) -> None:
        self.orientation_mapping_refresh()
        self.refresh_results()

    def refresh_results(self, *_args) -> None:
        self.mapping_group.setVisible(self.family.currentText() == "Orientation")
        entries = [entry for entry in self.result_registry.list_entries() if self._matches(entry)]
        self.workspace.set_results([
            FigureResult(
                entry.name,
                entry.data,
                image_kind="color" if np.asarray(entry.data).ndim == 3 else "intensity",
                scaling="linear",
            )
            for entry in entries[:6]
            if self._displayable(entry.data)
        ])

    def _matches(self, entry: ResultEntry) -> bool:
        category = entry.category.lower()
        family = self.family.currentText().lower()
        if family != "all" and not category.startswith(family):
            return False
        view = self.view.currentText()
        name = entry.name.lower()
        if view == "Final Results":
            return name in self.FINAL_NAMES.get("orientation" if category.startswith("orientation") else "strain", set())
        if view == "Quality Maps":
            return any(word in name for word in self.QUALITY_WORDS)
        return "process" in category or (
            name not in self.FINAL_NAMES.get(
                "orientation" if category.startswith("orientation") else "strain", set()
            ) and not any(word in name for word in self.QUALITY_WORDS)
        )

    @staticmethod
    def _displayable(data) -> bool:
        try:
            return np.asarray(data).ndim in {2, 3}
        except Exception:
            return False
