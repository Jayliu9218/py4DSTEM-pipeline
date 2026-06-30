import unittest

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea
from PySide6.QtWidgets import QSizePolicy

from app.theme import ACTION_BUTTON_MIN_HEIGHT, PARAM_ROW_HEIGHT
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.scientific_controls import (
    SCI_ACTION_ROW,
    SCI_CONTROLS_PANEL,
    SCI_DISABLED_BUTTON,
    SCI_PROPERTY_LABEL,
    SCI_PROPERTY_UNIT,
    SCI_PROPERTY_VALUE,
    SCI_SECTION,
    SCI_STATUS_MESSAGE,
    SCI_STATUS_ROW,
    ScientificControlsPanel,
    action_row,
    property_row,
    section,
    status_row,
)


class ScientificControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_section_applies_numbered_title_and_property_hooks(self) -> None:
        value = QLabel("value")
        panel = ScientificControlsPanel([
            section("Detector", [property_row("Mode", value)], number=3)
        ])

        self.assertEqual(panel.objectName(), SCI_CONTROLS_PANEL)
        scientific_section = panel.findChild(type(panel.layout().itemAt(0).widget()))
        self.assertEqual(scientific_section.objectName(), SCI_SECTION)
        self.assertEqual(scientific_section.title(), "3 Detector")
        self.assertEqual(
            scientific_section.findChildren(QLabel, SCI_PROPERTY_LABEL)[0].text(),
            "Mode",
        )
        value_cell = scientific_section.value_for_field(value)
        self.assertIsNotNone(value_cell)
        self.assertEqual(value_cell.objectName(), SCI_PROPERTY_VALUE)
        self.assertEqual(value_cell.property("rowParity"), "even")
        self.assertEqual(scientific_section.grid.columnMinimumWidth(0), 190)
        self.assertEqual(scientific_section.grid.columnStretch(1), 1)

    def test_constructor_sections_keep_input_order_before_stretch(self) -> None:
        panel = ScientificControlsPanel([
            section("First", []),
            section("Second", []),
            section("Third", []),
        ])

        layout = panel.layout()
        self.assertEqual(layout.itemAt(0).widget().title(), "First")
        self.assertEqual(layout.itemAt(1).widget().title(), "Second")
        self.assertEqual(layout.itemAt(2).widget().title(), "Third")
        self.assertIsNone(layout.itemAt(3).widget())

    def test_action_row_and_status_row_expose_shared_hooks(self) -> None:
        primary = QPushButton("Run")
        secondary = QPushButton("Cancel")
        secondary.setEnabled(False)
        actions = action_row(primary, secondary)
        status = status_row("Idle")

        self.assertEqual(actions.objectName(), SCI_ACTION_ROW)
        for button in (primary, secondary):
            self.assertEqual(button.minimumHeight(), ACTION_BUTTON_MIN_HEIGHT)
            self.assertEqual(button.maximumHeight(), ACTION_BUTTON_MIN_HEIGHT)
        self.assertTrue(secondary.property(SCI_DISABLED_BUTTON))
        self.assertEqual(status.objectName(), SCI_STATUS_ROW)
        self.assertEqual(status.findChild(QLabel).objectName(), SCI_STATUS_MESSAGE)

    def test_unit_aware_numeric_value_keeps_unit_next_to_value(self) -> None:
        control = NumericLineEdit(0, 100, 10, decimals=2, unit="um")

        self.assertGreaterEqual(control.line_edit.minimumWidth(), 90)
        self.assertEqual(control.unit_label.text(), "um")
        self.assertEqual(control.layout().spacing(), 6)
        self.assertEqual(control.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)

    def test_scientific_section_splits_numeric_value_and_unit_columns(self) -> None:
        control = NumericLineEdit(0, 1_000_000, 300_000, decimals=0, unit="V", integer=True)
        scientific_section = section("Plan", [property_row("Accelerating voltage", control)])

        value_cell = scientific_section.value_for_field(control)
        unit_cell = scientific_section.unit_for_field(control)

        self.assertIs(value_cell, control.line_edit)
        self.assertIs(unit_cell, control.unit_label)
        self.assertEqual(value_cell.objectName(), SCI_PROPERTY_VALUE)
        self.assertEqual(unit_cell.objectName(), SCI_PROPERTY_UNIT)
        self.assertEqual(scientific_section.grid.columnMinimumWidth(1), 90)
        self.assertEqual(scientific_section.grid.columnMinimumWidth(2), 48)

    def test_long_scientific_sections_scroll_parameters_after_five_visible_rows(self) -> None:
        run = QPushButton("Run")
        scientific_section = section("Many Parameters", [
            *(property_row(f"Parameter {index}", QLabel(str(index))) for index in range(1, 8)),
            property_row("", action_row(run)),
        ])

        scroll = scientific_section.findChild(QScrollArea, "ScientificParameterScroll")

        self.assertIsNotNone(scroll)
        self.assertEqual(scroll.maximumHeight(), 5 * PARAM_ROW_HEIGHT)
        self.assertIsNone(scroll.widget().findChild(QPushButton))
        scroll_grid = scroll.widget().layout()
        scrolled_labels = [
            item.widget().text()
            for row in range(scroll_grid.rowCount())
            if (item := scroll_grid.itemAtPosition(row, 0)) is not None
        ]
        self.assertEqual(scrolled_labels[:5], [f"Parameter {index}" for index in range(1, 6)])
        self.assertEqual(scrolled_labels[5:], ["Parameter 6", "Parameter 7"])


if __name__ == "__main__":
    unittest.main()
