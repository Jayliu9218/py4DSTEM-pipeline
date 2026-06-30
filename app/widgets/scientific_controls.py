from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.theme import ACTION_BUTTON_MIN_HEIGHT, GROUP_SPACING, PANEL_MARGIN_TIGHT, PARAM_ROW_HEIGHT
from app.widgets.numeric_line_edit import NumericLineEdit


SCI_CONTROLS_PANEL = "ScientificControlsPanel"
SCI_SECTION = "ScientificSection"
SCI_SECTION_HEADER = "ScientificSectionHeader"
SCI_SECTION_TITLE = "ScientificSectionTitle"
SCI_PROPERTY_GRID = "ScientificPropertyGrid"
SCI_PROPERTY_ROW = "ScientificPropertyRow"
SCI_PROPERTY_LABEL = "ScientificPropertyLabel"
SCI_PROPERTY_VALUE = "ScientificPropertyValue"
SCI_PROPERTY_UNIT = "ScientificPropertyUnit"
SCI_ACTION_ROW = "ScientificActionRow"
SCI_STATUS_ROW = "ScientificStatusRow"
SCI_STATUS_MESSAGE = "ScientificStatusMessage"
SCI_PRIMARY_BUTTON = "ScientificPrimaryButton"
SCI_SECONDARY_BUTTON = "ScientificSecondaryButton"
SCI_DISABLED_BUTTON = "ScientificDisabledButton"


class ScientificControlsPanel(QWidget):
    """Stacked, compact control panel for scientific workflow modules."""

    def __init__(self, sections: Iterable[QWidget] | None = None) -> None:
        super().__init__()
        self.setObjectName(SCI_CONTROLS_PANEL)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(GROUP_SPACING)
        self._layout.addStretch(1)
        if sections is not None:
            for section in sections:
                self.add_section(section)

    def add_section(self, section: QWidget) -> None:
        insert_at = max(self._layout.count() - 1, 0)
        self._layout.insertWidget(insert_at, section)


class ScientificSection(QGroupBox):
    """A property-grid section with optional numbering in the title."""

    PARAM_VISIBLE_ROWS = 5

    def __init__(
        self,
        title: str,
        rows: Iterable[tuple[str, QWidget]] | None = None,
        number: int | None = None,
    ) -> None:
        display_title = f"{number} {title}" if number is not None else title
        super().__init__(display_title)
        self.setObjectName(SCI_SECTION)
        self.setProperty("panelMode", "propertyGrid")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.grid = QGridLayout(self)
        self.grid.setObjectName(SCI_PROPERTY_GRID)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(0)
        self.grid.setVerticalSpacing(0)
        self.grid.setColumnMinimumWidth(0, 190)
        self.grid.setColumnMinimumWidth(1, 90)
        self.grid.setColumnMinimumWidth(2, 48)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 0)
        self._parameter_row = 0
        self._field_labels: dict[QWidget, QLabel] = {}
        self._field_values: dict[QWidget, QWidget] = {}
        self._field_units: dict[QWidget, QWidget] = {}
        self._parameter_scroll: QScrollArea | None = None
        self._parameter_scroll_grid: QGridLayout | None = None
        self._parameter_entries: list[tuple[QLabel, QWidget, QWidget | None]] = []
        if rows is not None:
            for label, widget in rows:
                self.add_row(label, widget)

    def add_row(self, label: str, widget: QWidget) -> None:
        if label:
            row_parity = "even" if self._parameter_row % 2 == 0 else "odd"
            label_widget = QLabel(label)
            label_widget.setObjectName(SCI_PROPERTY_LABEL)
            label_widget.setProperty("rowParity", row_parity)
            label_widget.setAutoFillBackground(True)
            label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value, unit = _property_cells(widget)
            _style_property_cell(value, row_parity)
            if unit is not None:
                _style_property_cell(unit, row_parity)
            self._field_labels[widget] = label_widget
            self._field_values[widget] = value
            if unit is not None:
                self._field_units[widget] = unit
            self._parameter_entries.append((label_widget, value, unit))
            self._parameter_row += 1
            self._place_parameter_row(label_widget, value, unit)
            return
        widget.setObjectName(widget.objectName() or SCI_ACTION_ROW)
        widget.setProperty("scientificAction", True)
        row = self.grid.rowCount()
        self.grid.addWidget(widget, row, 0, 1, 3)

    def label_for_field(self, widget: QWidget) -> QLabel | None:
        return self._field_labels.get(widget)

    def value_for_field(self, widget: QWidget) -> QWidget | None:
        return self._field_values.get(widget)

    def unit_for_field(self, widget: QWidget) -> QWidget | None:
        return self._field_units.get(widget)

    def _place_parameter_row(self, label: QLabel, value: QWidget, unit: QWidget | None) -> None:
        if len(self._parameter_entries) > self.PARAM_VISIBLE_ROWS:
            self._ensure_parameter_scroll()
            assert self._parameter_scroll_grid is not None
            self._add_parameter_widgets(self._parameter_scroll_grid, self._parameter_scroll_grid.rowCount(), label, value, unit)
            return
        self._add_parameter_widgets(self.grid, self.grid.rowCount(), label, value, unit)

    def _ensure_parameter_scroll(self) -> None:
        if self._parameter_scroll is not None:
            return
        entries = list(self._parameter_entries)
        for label, value, unit in entries:
            self.grid.removeWidget(label)
            label.setParent(None)
            self.grid.removeWidget(value)
            value.setParent(None)
            if unit is not None:
                self.grid.removeWidget(unit)
                unit.setParent(None)

        content = QWidget()
        content.setObjectName("ScientificParameterScrollContent")
        scroll_grid = QGridLayout(content)
        scroll_grid.setObjectName(SCI_PROPERTY_GRID)
        scroll_grid.setContentsMargins(0, 0, 0, 0)
        scroll_grid.setHorizontalSpacing(0)
        scroll_grid.setVerticalSpacing(0)
        scroll_grid.setColumnMinimumWidth(0, 190)
        scroll_grid.setColumnMinimumWidth(1, 90)
        scroll_grid.setColumnMinimumWidth(2, 48)
        scroll_grid.setColumnStretch(0, 0)
        scroll_grid.setColumnStretch(1, 1)
        scroll_grid.setColumnStretch(2, 0)
        for row, (label, value, unit) in enumerate(entries):
            self._add_parameter_widgets(scroll_grid, row, label, value, unit)

        scroll = QScrollArea()
        scroll.setObjectName("ScientificParameterScroll")
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMaximumHeight(PARAM_ROW_HEIGHT * self.PARAM_VISIBLE_ROWS)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.grid.addWidget(scroll, 0, 0, 1, 3)
        self._parameter_scroll = scroll
        self._parameter_scroll_grid = scroll_grid

    @staticmethod
    def _add_parameter_widgets(
        grid: QGridLayout,
        row: int,
        label: QLabel,
        value: QWidget,
        unit: QWidget | None,
    ) -> None:
        grid.addWidget(label, row, 0)
        if unit is None:
            grid.addWidget(value, row, 1, 1, 2)
        else:
            grid.addWidget(value, row, 1)
            grid.addWidget(unit, row, 2)


def section(title: str, rows: Iterable[tuple[str, QWidget]], number: int | None = None) -> ScientificSection:
    return ScientificSection(title, rows, number)


def property_row(label: str, widget: QWidget) -> tuple[str, QWidget]:
    return label, widget


def action_row(*buttons: QWidget) -> QWidget:
    container = QWidget()
    container.setObjectName(SCI_ACTION_ROW)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, PANEL_MARGIN_TIGHT, 0, PANEL_MARGIN_TIGHT)
    layout.setSpacing(GROUP_SPACING)
    for index, button in enumerate(buttons):
        if isinstance(button, QPushButton):
            button.setObjectName(button.objectName() or (SCI_PRIMARY_BUTTON if index == 0 else SCI_SECONDARY_BUTTON))
            button.setMinimumHeight(ACTION_BUTTON_MIN_HEIGHT)
            button.setMaximumHeight(ACTION_BUTTON_MIN_HEIGHT)
            button.setProperty(SCI_DISABLED_BUTTON, not button.isEnabled())
        layout.addWidget(button)
    return container


def status_row(label_or_message: str | QLabel) -> QWidget:
    container = QWidget()
    container.setObjectName(SCI_STATUS_ROW)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, PANEL_MARGIN_TIGHT, 0, PANEL_MARGIN_TIGHT)
    if isinstance(label_or_message, QLabel):
        message = label_or_message
    else:
        message = QLabel(label_or_message)
    message.setObjectName(SCI_STATUS_MESSAGE)
    message.setWordWrap(True)
    layout.addWidget(message)
    return container


def _property_cells(widget: QWidget) -> tuple[QWidget, QWidget | None]:
    if isinstance(widget, NumericLineEdit):
        editor = widget.line_edit
        unit = widget.unit_label
        editor.setObjectName(SCI_PROPERTY_VALUE)
        editor.setMinimumWidth(90)
        editor.setMaximumWidth(16777215)
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        unit.setObjectName(SCI_PROPERTY_UNIT)
        unit.setMinimumWidth(48 if unit.text() else 0)
        unit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        unit.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return editor, unit
    widget.setObjectName(widget.objectName() or SCI_PROPERTY_VALUE)
    widget.setProperty("scientificValue", True)
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Expanding, widget.sizePolicy().verticalPolicy())
    if isinstance(widget, QLineEdit):
        widget.setMinimumWidth(90)
    return widget, None


def _style_property_cell(widget: QWidget, row_parity: str) -> None:
    widget.setProperty("rowParity", row_parity)
    widget.setAutoFillBackground(True)
