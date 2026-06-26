from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.theme import (
    ACTION_BUTTON_MIN_HEIGHT,
    GROUP_SPACING,
    PANEL_MARGIN,
    PANEL_MARGIN_TIGHT,
    PARAM_TABLE_HEIGHT,
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
    SHOW_ROUTE_ARROWS = False

    STATE_COLORS = {
        "Completed": "#4caf50",
        "Warning": "#ff9800",
        "Ready": "#757575",
        "Disabled": "#555555",
    }

    def __init__(self) -> None:
        super().__init__()
        self.modules: list[RouteModule] = []
        self.buttons: dict[str, QToolButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN_TIGHT, PANEL_MARGIN, PANEL_MARGIN_TIGHT)
        self.layout.setSpacing(GROUP_SPACING)

    def set_modules(self, modules: list[RouteModule]) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.buttons.clear()
        self.modules = modules
        for index, module in enumerate(modules):
            button = QToolButton()
            button.setObjectName("routeButton")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            # Long workflow names must shrink with the central workspace instead
            # of forcing neighbouring route buttons and arrows to overlap.
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            # Surface stub modules (implemented=False) as greyed-out and non-clickable.
            if not module.implemented:
                button.setProperty("implemented", "false")
                button.setToolTip(f"{module.title} — coming soon")
                button.setEnabled(False)
            else:
                button.setProperty("implemented", "true")
                button.setToolTip("")
                button.setEnabled(True)
            button.clicked.connect(lambda _checked=False, key=module.key: self.module_selected.emit(key))
            self.group.addButton(button)
            self.buttons[module.key] = button
            self.layout.addWidget(button, 1)
            if self.SHOW_ROUTE_ARROWS and index < len(modules) - 1:
                arrow = QLabel("›")
                arrow.setObjectName("routeArrow")
                self.layout.addWidget(arrow)

    def update_states(self, states: dict[str, str], current_key: str) -> None:
        for module in self.modules:
            button = self.buttons[module.key]
            state = states.get(module.key, "Ready")
            color = self.STATE_COLORS.get(state, self.STATE_COLORS["Ready"])
            button.setText(f"●  {module.title}")
            is_selected = module.key == current_key
            # Disabled buttons (no data yet, or stub modules) are non-interactive.
            is_disabled = state == "Disabled" or not module.implemented
            button.setEnabled(not is_disabled)
            # Non-selected buttons show status-colored text; selected buttons
            # use the QSS white text so it's readable on the highlighted background.
            if is_selected:
                button.setStyleSheet("")
            else:
                button.setStyleSheet(f"QToolButton#routeButton {{ color: {color}; }}")
            button.setChecked(is_selected)


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
        layout.setContentsMargins(0, 0, 0, 0)
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
        self.setObjectName("moduleControlPanel")
        self.title = QLabel("Data Setup")
        self.title.setObjectName("moduleTitle")
        self.controls_stack = QStackedWidget()
        self.controls_stack.setObjectName("moduleControlsSurface")
        self._controls: dict[int, QWidget] = {}
        self._placeholder = QLabel("Select a module to inspect its parameters.")
        self._placeholder.setObjectName("moduleControlsContent")
        self._placeholder.setWordWrap(True)
        self.controls_stack.addWidget(self._placeholder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN)
        layout.setSpacing(GROUP_SPACING)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        layout.addLayout(header)
        layout.addWidget(self.controls_stack, 1)

    def set_module(self, module: RouteModule, controls: QWidget | None) -> None:
        self.title.setText(module.title)
        if controls is None:
            self.controls_stack.setCurrentWidget(self._placeholder)
            return
        self._prepare_controls_layout(controls)
        identity = id(controls)
        if identity not in self._controls:
            if isinstance(controls, QScrollArea):
                scroll = controls
                content = scroll.widget()
            else:
                scroll = QScrollArea()
                controls.setMinimumSize(0, 0)
                controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                scroll.setWidget(controls)
                content = controls
            scroll.setObjectName("moduleControlsScroll")
            if content is not None and not content.objectName():
                content.setObjectName("moduleControlsContent")
            scroll.setWidgetResizable(True)
            scroll.setMinimumSize(0, 0)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            self._controls[identity] = scroll
            self.controls_stack.addWidget(scroll)
        self.controls_stack.setCurrentWidget(self._controls[identity])

    @staticmethod
    def _prepare_controls_layout(controls: QWidget) -> None:
        content = controls.widget() if isinstance(controls, QScrollArea) else controls
        layout = content.layout() if content is not None else None
        if layout is None:
            return
        layout.setContentsMargins(PANEL_MARGIN + 1, PANEL_MARGIN, PANEL_MARGIN + 1, PANEL_MARGIN)
        layout.setSpacing(GROUP_SPACING)
        if isinstance(content, QGroupBox):
            ModuleControlPanel._apply_table_form(layout, -1, content)
        for table in content.findChildren(QTableWidget):
            table.setFixedHeight(PARAM_TABLE_HEIGHT)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        for button in content.findChildren(QPushButton):
            button.setMinimumHeight(ACTION_BUTTON_MIN_HEIGHT)
            button.setMaximumHeight(ACTION_BUTTON_MIN_HEIGHT)
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                ModuleControlPanel._apply_table_form(layout, index, widget)

    @staticmethod
    def _apply_table_form(parent_layout, index: int, group: QGroupBox) -> None:
        """Make a parameter group fixed-height and table-like.

        The module panel owns the single vertical scroll area, so groups stay
        intact and page-specific form policies remain unchanged.
        """
        # Apply table-like layout policies to every QFormLayout in this box.
        for form in group.findChildren(QFormLayout):
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN_TIGHT, PANEL_MARGIN, PANEL_MARGIN_TIGHT)
            if ModuleControlPanel._labeled_parameter_count(form) > 4:
                form.setContentsMargins(0, 0, 0, 0)
                form.setVerticalSpacing(0)
                form.setHorizontalSpacing(0)
                ModuleControlPanel._apply_property_grid_rows(form)
            else:
                form.setVerticalSpacing(6)
                form.setHorizontalSpacing(10)
        if group.objectName() != "ScientificSection":
            group.setObjectName("paramForm")

        # Keep groups compact, but allow wrapped status/warning labels to grow
        # after a calculation updates their text.
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        group.setMinimumHeight(0)
        group.setMaximumHeight(16777215)

    @staticmethod
    def _form_widget(form: QFormLayout, row: int, role: QFormLayout.ItemRole) -> QWidget | None:
        item = form.itemAt(row, role)
        return item.widget() if item is not None else None

    @staticmethod
    def _labeled_parameter_count(form: QFormLayout) -> int:
        total = 0
        for row in range(form.rowCount()):
            label = ModuleControlPanel._form_widget(form, row, QFormLayout.LabelRole)
            field = ModuleControlPanel._form_widget(form, row, QFormLayout.FieldRole)
            if isinstance(label, QLabel) and label.text().strip() and field is not None:
                total += 1
        return total

    @staticmethod
    def _apply_property_grid_rows(form: QFormLayout) -> None:
        parameter_row = 0
        for row in range(form.rowCount()):
            label = ModuleControlPanel._form_widget(form, row, QFormLayout.LabelRole)
            field = ModuleControlPanel._form_widget(form, row, QFormLayout.FieldRole)
            if isinstance(label, QLabel) and label.text().strip() and field is not None:
                row_parity = "even" if parameter_row % 2 == 0 else "odd"
                label.setObjectName("propertyGridLabel")
                label.setProperty("rowParity", row_parity)
                label.setAutoFillBackground(True)
                label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                field.setObjectName(field.objectName() or "propertyGridValue")
                field.setProperty("rowParity", row_parity)
                field.setAutoFillBackground(True)
                parameter_row += 1
            elif field is not None:
                field.setObjectName(field.objectName() or "propertyGridAction")


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
        layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN_TIGHT, PANEL_MARGIN, PANEL_MARGIN_TIGHT)
        layout.addStretch()

        layout.addWidget(QLabel("Analysis Route"), alignment=Qt.AlignRight)
        layout.addWidget(self.structure)

        layout.addSpacing(12)

        layout.addWidget(QLabel("Target"), alignment=Qt.AlignRight)
        layout.addWidget(self.goal)

        layout.addStretch()

        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("exportButton")
        self.export_button.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.export_button)

    def set_goals(self, goals: list[str]) -> None:
        current = self.goal.currentText()
        self.goal.blockSignals(True)
        self.goal.clear()
        self.goal.addItems(goals)
        if current in goals:
            self.goal.setCurrentText(current)
        self.goal.setEnabled(len(goals) > 1)
        self.goal.blockSignals(False)
