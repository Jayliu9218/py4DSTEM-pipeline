from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.image_viewer import ImageViewer


STATUS_COLORS = {
    "Completed": "#18794e",
    "Current": "#1769aa",
    "Ready": "#4d6b82",
    "Locked": "#8b96a1",
    "Warning": "#b26a00",
}


@dataclass(frozen=True)
class RouteModule:
    key: str
    title: str
    page_key: str
    requirements: str
    output_target: str
    state_step: str | None = None
    prerequisite: str | None = None
    implemented: bool = True


class TechnicalRouteBar(QWidget):
    module_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.modules: list[RouteModule] = []
        self.buttons: dict[str, QToolButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 6, 10, 6)
        self.layout.setSpacing(5)

    def set_modules(self, modules: list[RouteModule]) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.buttons.clear()
        self.modules = modules
        for index, module in enumerate(modules):
            button = QToolButton()
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda _checked=False, key=module.key: self.module_selected.emit(key))
            self.group.addButton(button)
            self.buttons[module.key] = button
            self.layout.addWidget(button, 1)
            if index < len(modules) - 1:
                arrow = QLabel("›")
                arrow.setObjectName("routeArrow")
                self.layout.addWidget(arrow)

    def update_states(self, states: dict[str, str], current_key: str) -> None:
        for module in self.modules:
            state = "Current" if module.key == current_key else states.get(module.key, "Locked")
            button = self.buttons[module.key]
            button.setText(f"{module.title}\n{state}")
            button.setProperty("routeState", state)
            button.setChecked(module.key == current_key)


class DataStatePanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Data State")
        layout = QVBoxLayout(self)
        self.source = QLabel("Source: Not loaded")
        self.target = QLabel("Target: Not assigned")
        self.selection = QLabel("Selection: None")
        self.readiness = QLabel("Readiness: Waiting for DataCube")
        for label in (self.source, self.target, self.selection, self.readiness):
            label.setWordWrap(True)
            layout.addWidget(label)

    def update_state(
        self,
        source: str | None,
        target: str | None,
        selection: str | None,
        ready: bool,
    ) -> None:
        self.source.setText(f"Source: {source or 'Not loaded'}")
        self.target.setText(f"Target: {target or 'Not assigned'}")
        self.selection.setText(f"Selection: {selection or 'None'}")
        self.readiness.setText(
            "Readiness: Data setup available" if ready else "Readiness: Waiting for DataCube"
        )
        self.readiness.setProperty("ready", ready)


class MultiViewWorkspace(QWidget):
    def __init__(self, scan_viewer: ImageViewer, diffraction_viewer: ImageViewer) -> None:
        super().__init__()
        self.viewers = [
            scan_viewer,
            diffraction_viewer,
            ImageViewer(),
            ImageViewer(),
        ]
        self.titles = [
            "Virtual / Real-space Image",
            "Diffraction Pattern",
            "Analysis Result",
            "Diagnostic / Comparison",
        ]
        self.layout_mode = QComboBox()
        self.layout_mode.addItems(["Single View", "Split View", "Quad View"])
        self.layout_mode.setCurrentText("Split View")
        self.layout_mode.currentTextChanged.connect(self._apply_layout)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(5)
        self.panels = [self._panel(title, viewer) for title, viewer in zip(self.titles, self.viewers)]
        header = QHBoxLayout()
        title = QLabel("Main Viewer")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QLabel("Layout"))
        header.addWidget(self.layout_mode)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addLayout(self.grid, 1)
        self._apply_layout("Split View")

    def _panel(self, title: str, viewer: ImageViewer) -> QWidget:
        panel = QFrame()
        panel.setObjectName("viewerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        label = QLabel(title)
        label.setObjectName("viewerTitle")
        layout.addWidget(label)
        layout.addWidget(viewer, 1)
        return panel

    def _apply_layout(self, mode: str) -> None:
        for panel in self.panels:
            self.grid.removeWidget(panel)
            panel.setVisible(False)
        positions = [(0, 0)]
        if mode == "Split View":
            positions = [(0, 0), (0, 1)]
        elif mode == "Quad View":
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for panel, position in zip(self.panels, positions):
            self.grid.addWidget(panel, *position)
            panel.setVisible(True)
        for index in range(2):
            self.grid.setRowStretch(index, 1)
            self.grid.setColumnStretch(index, 1)


class ModuleControlPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.title = QLabel("Data Setup")
        self.title.setObjectName("moduleTitle")
        self.status = QLabel("Current")
        self.status.setObjectName("statusBadge")
        self.controls_host = QVBoxLayout()
        self.controls_host.setContentsMargins(0, 0, 0, 0)
        self.controls_host.addWidget(QLabel("Select a module to inspect its parameters."))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.status)
        layout.addLayout(header)
        layout.addLayout(self.controls_host, 1)

    def set_module(self, module: RouteModule, status: str, controls: QWidget | None) -> None:
        self.title.setText(module.title)
        self.status.setText(status)
        while self.controls_host.count():
            item = self.controls_host.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        if controls is None:
            placeholder = QLabel(
                "This module is represented in the dependency graph. "
                "Its calculation action can be connected here without changing the main layout."
            )
            placeholder.setWordWrap(True)
            self.controls_host.addWidget(placeholder)
        else:
            self.controls_host.addWidget(controls)


class ProjectToolbar(QWidget):
    structure_changed = Signal(str)
    goal_changed = Signal(str)
    project_clicked = Signal()
    load_clicked = Signal()
    save_clicked = Signal()
    export_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("projectToolbar")
        font = self.font()
        font.setPointSize(font.pointSize() + 2)
        self.setFont(font)
        self.structure = QComboBox()
        self.structure.addItems(["Crystalline", "Amorphous", "Mixed / Nanocrystalline"])
        self.goal = QComboBox()
        self.structure.currentTextChanged.connect(self.structure_changed)
        self.goal.currentTextChanged.connect(self.goal_changed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        for text, signal in [
            ("Project", self.project_clicked),
            ("Load Data", self.load_clicked),
        ]:
            button = QPushButton(text)
            button.clicked.connect(signal)
            layout.addWidget(button)
        layout.addWidget(QLabel("Structure Type"))
        layout.addWidget(self.structure)
        layout.addWidget(QLabel("Analysis Goal"))
        layout.addWidget(self.goal, 1)
        save = QPushButton("Save")
        save.clicked.connect(self.save_clicked)
        export = QPushButton("Export")
        export.clicked.connect(self.export_clicked)
        layout.addWidget(save)
        layout.addWidget(export)

    def set_goals(self, goals: list[str]) -> None:
        current = self.goal.currentText()
        self.goal.blockSignals(True)
        self.goal.clear()
        self.goal.addItems(goals)
        if current in goals:
            self.goal.setCurrentText(current)
        self.goal.blockSignals(False)
