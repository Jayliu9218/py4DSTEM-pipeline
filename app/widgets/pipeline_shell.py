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

from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.image_viewer import ImageViewer


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
            button = self.buttons[module.key]
            button.setText(module.title)
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
        self.scan_viewer = scan_viewer
        self.diffraction_viewer = diffraction_viewer
        self.workspace = AdaptiveImageWorkspace()
        self.workspace.set_layout("2")
        self.clear_results()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.workspace, 1)

    def clear_results(self) -> None:
        self.workspace.set_results([
            FigureResult("Virtual / Real-space Image", [[0]], key="scan", viewer=self.scan_viewer),
            FigureResult("Diffraction Pattern", [[0]], key="diffraction", viewer=self.diffraction_viewer),
            FigureResult("Analysis Result", [[0]], key="analysis"),
            FigureResult("Diagnostic / Comparison", [[0]], key="diagnostic"),
        ])
        self.scan_viewer.clear("Mean real-space image / virtual bright field preview")
        self.diffraction_viewer.clear("No DataCube loaded.")

    def update_image(self, key: str, title: str, image, image_kind: str = "intensity") -> None:
        self.workspace.update_result(key, FigureResult(title, image, key=key, image_kind=image_kind))


class ModuleControlPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.title = QLabel("Data Setup")
        self.title.setObjectName("moduleTitle")
        self.controls_stack = QStackedWidget()
        self._controls: dict[int, QWidget] = {}
        self._placeholder = QLabel("Select a module to inspect its parameters.")
        self._placeholder.setWordWrap(True)
        self.controls_stack.addWidget(self._placeholder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        layout.addLayout(header)
        layout.addWidget(self.controls_stack, 1)

    def set_module(self, module: RouteModule, controls: QWidget | None) -> None:
        self.title.setText(module.title)
        if controls is None:
            self.controls_stack.setCurrentWidget(self._placeholder)
            return
        identity = id(controls)
        if identity not in self._controls:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMinimumSize(0, 0)
            scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            controls.setMinimumSize(0, 0)
            scroll.setWidget(controls)
            self._controls[identity] = scroll
            self.controls_stack.addWidget(scroll)
        self.controls_stack.setCurrentWidget(self._controls[identity])


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
        self.structure = QComboBox()
        self.structure.addItems(["Crystalline / Bragg-based", "Amorphous / Diffuse-scattering", "Phase Retrieval / Ptychography"])
        self.goal = QComboBox()
        self.structure.currentTextChanged.connect(self.structure_changed)
        self.goal.currentTextChanged.connect(self.goal_changed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.addStretch()

        layout.addWidget(QLabel("Analysis Route"), alignment=Qt.AlignRight)
        layout.addWidget(self.structure)

        layout.addSpacing(20)

        layout.addWidget(QLabel("Target"), alignment=Qt.AlignRight)
        layout.addWidget(self.goal)

        layout.addStretch()

    def set_goals(self, goals: list[str]) -> None:
        current = self.goal.currentText()
        self.goal.blockSignals(True)
        self.goal.clear()
        self.goal.addItems(goals)
        if current in goals:
            self.goal.setCurrentText(current)
        self.goal.blockSignals(False)
