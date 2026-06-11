import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult, FigurePanel


class AdaptiveImageWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def results(self, count: int) -> list[FigureResult]:
        return [FigureResult(str(index), np.ones((4, 4))) for index in range(count)]

    def test_automatic_capacity_matches_figure_count(self) -> None:
        self.assertEqual([AdaptiveImageWorkspace.automatic_capacity(i) for i in range(1, 7)], [1, 2, 4, 4, 6, 6])

    def test_more_than_six_results_are_paginated(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.set_results(self.results(8))
        self.assertEqual(workspace.page_capacity(), 6)
        self.assertEqual(workspace.page_count(), 2)
        self.assertEqual(len(workspace.visible_results()), 6)
        workspace.set_page(1)
        self.assertEqual(len(workspace.visible_results()), 2)

    def test_manual_override_and_minimum_panel_size(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.layout_choice.setCurrentText("2")
        workspace.set_results(self.results(4))
        self.assertEqual(workspace.page_capacity(), 2)
        self.assertTrue(all(panel.minimumWidth() >= FigurePanel.MINIMUM_WIDTH for panel in workspace.panels))

    def test_layout_switch_resets_positions_and_obsolete_stretch(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.set_results(self.results(6))
        expected = {
            "6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            "1": [(0, 0)],
            "2": [(0, 0), (0, 1)],
            "4": [(0, 0), (0, 1), (1, 0), (1, 1)],
        }
        original_panels = dict(workspace._panels_by_key)
        for layout in ("6", "1", "2", "4", "6"):
            workspace.set_layout(layout)
            positions = []
            for panel in workspace.panels:
                row, column, _row_span, _column_span = workspace.grid.getItemPosition(
                    workspace.grid.indexOf(panel)
                )
                positions.append((row, column))
            self.assertEqual(positions, expected[layout])
            rows, columns = workspace.grid_shape(int(layout))
            self.assertTrue(all(workspace.grid.columnStretch(index) == 0 for index in range(columns, 6)))
            self.assertTrue(all(workspace.grid.rowStretch(index) == 0 for index in range(rows, 6)))
        self.assertEqual(original_panels, workspace._panels_by_key)

    def test_append_update_clear_and_grid_state(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.update_result("live", FigureResult("Live A", np.ones((2, 2))))
        workspace.update_result("live", FigureResult("Live B", np.zeros((2, 2))))
        self.assertEqual(len(workspace.results), 1)
        workspace.append_result(FigureResult("Run", np.ones((2, 2))))
        self.assertEqual(len(workspace.results), 2)
        workspace.set_layout("1")
        workspace.set_page(1)
        state = workspace.grid_state()
        other = AdaptiveImageWorkspace()
        other.set_results(self.results(2))
        other.restore_grid_state(state)
        self.assertEqual(other.grid_state(), state)
        workspace.clear_results()
        self.assertEqual(workspace.results, [])


if __name__ == "__main__":
    unittest.main()
