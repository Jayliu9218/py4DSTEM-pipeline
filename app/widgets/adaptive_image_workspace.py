from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.image_viewer import ImageViewer


@dataclass(frozen=True)
class FigureResult:
    title: str
    image: Any
    diagnostic: str = ""
    image_kind: str = "intensity"
    levels: tuple[float, float] | None = None
    points: np.ndarray | None = None
    overlay: dict[str, float | str] | None = None
    bragg_sampling_provider: Callable[[int], np.ndarray] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    key: str = ""
    viewer: ImageViewer | None = None


class FigurePanel(QFrame):
    MINIMUM_WIDTH = 300
    MINIMUM_HEIGHT = 240

    def __init__(self, result: FigureResult) -> None:
        super().__init__()
        self.result = result
        self.setMinimumSize(self.MINIMUM_WIDTH, self.MINIMUM_HEIGHT)
        self.viewer = result.viewer or ImageViewer(result.image_kind)
        self.title_label = QLabel(result.title)
        self.title_label.setObjectName("viewerTitle")
        self.diagnostic_label = QLabel(result.diagnostic)
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.setVisible(bool(result.diagnostic))
        maximize = QPushButton("Maximize")
        maximize.clicked.connect(self._show_maximized)
        header = QHBoxLayout()
        header.addWidget(self.title_label, 1)
        header.addWidget(maximize)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(header)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(self.diagnostic_label)
        self.update_result(result)

    def update_result(self, result: FigureResult) -> None:
        self.result = result
        self.title_label.setText(result.title)
        self.diagnostic_label.setText(result.diagnostic)
        self.diagnostic_label.setVisible(bool(result.diagnostic))
        self._render(self.viewer)

    def _render(self, viewer: ImageViewer) -> None:
        viewer.clear_points()
        viewer.set_image(np.asarray(self.result.image), levels=self.result.levels)
        if self.result.points is not None and len(self.result.points):
            viewer.set_points(self.result.points[:, 0], self.result.points[:, 1], size=7)
        if self.result.bragg_sampling_provider is not None:
            viewer.set_bragg_sampling_provider(self.result.bragg_sampling_provider)
        overlay = self.result.overlay or {}
        if overlay.get("kind") == "circle":
            viewer.set_circle_overlay(
                float(overlay.get("x", 0)),
                float(overlay.get("y", 0)),
                float(overlay.get("r", 0)),
                color="r",
            )
        elif overlay.get("kind") == "ring":
            viewer.add_ring_overlay(
                float(overlay.get("x", 0)),
                float(overlay.get("y", 0)),
                float(overlay.get("inner_radius", 0)),
                float(overlay.get("outer_radius", 0)),
                color="r",
            )

    def _show_maximized(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self.result.title)
        dialog.resize(1100, 800)
        viewer = ImageViewer(self.result.image_kind)
        self._render(viewer)
        layout = QVBoxLayout(dialog)
        layout.addWidget(viewer)
        dialog.exec()


class AdaptiveImageWorkspace(QWidget):
    LAYOUT_CAPACITY = {"Auto": 0, "1": 1, "2": 2, "4": 4, "6": 6}

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.results: list[FigureResult] = []
        self._result_keys: list[str] = []
        self._panels_by_key: dict[str, FigurePanel] = {}
        self.panels: list[FigurePanel] = []
        self.current_page = 0
        self._last_resize = (-1, -1)
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(list(self.LAYOUT_CAPACITY))
        self.layout_choice.currentTextChanged.connect(self._layout_changed)
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.previous_button.clicked.connect(lambda: self.set_page(self.current_page - 1))
        self.next_button.clicked.connect(lambda: self.set_page(self.current_page + 1))
        self.page_label = QLabel("Page 0 / 0")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Grid"))
        controls.addWidget(self.layout_choice)
        controls.addStretch(1)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.page_label)
        controls.addWidget(self.next_button)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(self.grid, 1)
        self._render()

    @staticmethod
    def automatic_capacity(count: int) -> int:
        if count <= 1:
            return 1
        if count == 2:
            return 2
        if count <= 4:
            return 4
        return 6

    @staticmethod
    def grid_shape(capacity: int) -> tuple[int, int]:
        return {1: (1, 1), 2: (1, 2), 4: (2, 2), 6: (2, 3)}[capacity]

    def set_results(self, results: list[FigureResult]) -> None:
        normalized = [self._with_key(result, f"result-{index}") for index, result in enumerate(results)]
        keep = {result.key for result in normalized}
        for key in list(self._panels_by_key):
            if key not in keep:
                panel = self._panels_by_key.pop(key)
                self.grid.removeWidget(panel)
                panel.deleteLater()
        self.results = normalized
        self._result_keys = [result.key for result in normalized]
        self.current_page = 0
        self._render()

    def append_result(self, result: FigureResult) -> str:
        result = self._with_key(result, f"history-{uuid4().hex}")
        if result.key in self._result_keys:
            result = self._with_key(result, f"{result.key}-{uuid4().hex}")
        self.results.append(result)
        self._result_keys.append(result.key)
        self._render()
        return result.key

    def update_result(self, key: str, result: FigureResult) -> None:
        result = self._with_key(result, key)
        if key in self._result_keys:
            index = self._result_keys.index(key)
            self.results[index] = result
        else:
            self._result_keys.append(key)
            self.results.append(result)
        panel = self._panels_by_key.get(key)
        if panel is not None:
            panel.update_result(result)
        self._render()

    def clear_results(self) -> None:
        self.set_results([])

    def clear(self) -> None:
        self.clear_results()

    def set_layout(self, layout: str) -> None:
        if layout not in self.LAYOUT_CAPACITY:
            raise ValueError(f"Unsupported grid layout: {layout}")
        self.layout_choice.setCurrentText(layout)

    def grid_state(self) -> dict[str, object]:
        return {"layout": self.layout_choice.currentText(), "page": self.current_page}

    def restore_grid_state(self, state: dict[str, object]) -> None:
        layout = str(state.get("layout", "Auto"))
        page = int(state.get("page", 0))
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentText(layout if layout in self.LAYOUT_CAPACITY else "Auto")
        self.layout_choice.blockSignals(False)
        self.current_page = max(page, 0)
        self._render()

    def set_page(self, page: int) -> None:
        page_count = self.page_count()
        self.current_page = min(max(page, 0), max(page_count - 1, 0))
        self._render()

    def page_capacity(self) -> int:
        requested = self.LAYOUT_CAPACITY[self.layout_choice.currentText()]
        capacity = requested or self.automatic_capacity(len(self.results))
        width_capacity = max(self.width() // FigurePanel.MINIMUM_WIDTH, 1)
        height_capacity = max((self.height() - 45) // FigurePanel.MINIMUM_HEIGHT, 1)
        fitting = width_capacity * height_capacity
        supported = [value for value in (1, 2, 4, 6) if value <= fitting]
        return min(capacity, max(supported, default=1))

    def page_count(self) -> int:
        return ceil(len(self.results) / self.page_capacity()) if self.results else 0

    def visible_results(self) -> list[FigureResult]:
        capacity = self.page_capacity()
        start = self.current_page * capacity
        return self.results[start : start + capacity]

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        size = (event.size().width(), event.size().height())
        if self.isVisible() and size != self._last_resize:
            self._last_resize = size
            self._render()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        self._render()

    def refresh_layout(self) -> None:
        if self.isVisible():
            self._render()

    def _layout_changed(self, _text: str) -> None:
        self.current_page = 0
        self._render()

    def _render(self) -> None:
        for panel in self._panels_by_key.values():
            self.grid.removeWidget(panel)
            panel.hide()
        for row in range(6):
            self.grid.setRowStretch(row, 0)
            self.grid.setRowMinimumHeight(row, 0)
        for column in range(6):
            self.grid.setColumnStretch(column, 0)
            self.grid.setColumnMinimumWidth(column, 0)
        self.panels = []
        capacity = self.page_capacity()
        pages = self.page_count()
        if pages and self.current_page >= pages:
            self.current_page = pages - 1
        elif not pages:
            self.current_page = 0
        rows, columns = self.grid_shape(capacity)
        for index, result in enumerate(self.visible_results()):
            panel = self._panels_by_key.get(result.key)
            if panel is None:
                panel = FigurePanel(result)
                self._panels_by_key[result.key] = panel
            elif panel.result is not result:
                panel.update_result(result)
            self.panels.append(panel)
            self.grid.addWidget(panel, index // columns, index % columns)
            panel.show()
        for row in range(rows):
            self.grid.setRowStretch(row, 1)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)
        self.page_label.setText(f"Page {self.current_page + 1 if pages else 0} / {pages}")
        self.previous_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page + 1 < pages)

    @staticmethod
    def _with_key(result: FigureResult, fallback: str) -> FigureResult:
        if result.key:
            return result
        values = dict(result.__dict__)
        values["key"] = fallback
        return FigureResult(**values)
