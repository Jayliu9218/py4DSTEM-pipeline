import unittest

import numpy as np
from PySide6.QtWidgets import QApplication
from app.widgets.image_viewer import ImageViewer
from app.widgets.rgb_image_viewer import RgbImageViewer

from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult, FigurePanel


class AdaptiveImageWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def results(self, count: int) -> list[FigureResult]:
        return [FigureResult(str(index), np.ones((4, 4))) for index in range(count)]

    def test_automatic_capacity_matches_figure_count(self) -> None:
        self.assertEqual([AdaptiveImageWorkspace.automatic_capacity(i) for i in range(1, 7)], [1, 2, 4, 4, 6, 6])

    def test_workspace_owns_data_setup_style_content_margin(self) -> None:
        workspace = AdaptiveImageWorkspace()
        margins = workspace.layout().contentsMargins()
        self.assertEqual(
            (margins.left(), margins.top(), margins.right(), margins.bottom()),
            (0, 8, 0, 8),
        )

    def test_grid_toolbar_controls_have_stable_default_geometry(self) -> None:
        workspace = AdaptiveImageWorkspace()

        self.assertEqual(workspace.layout_choice.width(), 66)
        self.assertEqual(workspace.reset_button.width(), 126)
        self.assertEqual(workspace.previous_button.width(), 112)
        self.assertEqual(workspace.page_label.width(), 100)
        self.assertEqual(workspace.next_button.width(), 88)

    def test_set_results_is_capped_at_six(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.set_results(self.results(8))
        self.assertEqual(workspace.page_capacity(), 6)
        self.assertEqual(workspace.page_count(), 1)
        self.assertEqual(len(workspace.visible_results()), 6)

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

    def test_layout_change_signal_and_auto_lock(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.resize(1200, 800)
        workspace.set_results(self.results(4))
        changes = []
        workspace.layout_changed.connect(changes.append)

        locked = workspace.lock_auto_layout()

        self.assertEqual(locked, "4")
        self.assertEqual(workspace.layout_choice.currentText(), "4")
        self.assertEqual(changes, ["4"])

    def test_batch_append_keeps_or_replaces_complete_batch(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.append_results(self.results(4))
        workspace.append_results(self.results(2))
        self.assertEqual(len(workspace.results), 6)
        replacement = [FigureResult(f"new-{index}", np.ones((2, 2))) for index in range(3)]
        workspace.append_results(replacement)
        self.assertEqual([result.title for result in workspace.results], ["new-0", "new-1", "new-2"])

    def test_large_batch_keeps_first_six(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.append_results(self.results(8))
        self.assertEqual([result.title for result in workspace.results], [str(index) for index in range(6)])

    def test_new_fixed_slot_obeys_cap_and_existing_slot_does_not_grow(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.set_results(self.results(6))
        workspace.update_result("new-live", FigureResult("New Live", np.ones((2, 2))))
        self.assertEqual(len(workspace.results), 1)
        workspace.update_result("new-live", FigureResult("Updated Live", np.zeros((2, 2))))
        self.assertEqual(len(workspace.results), 1)

    def test_reset_grid_clears_images_preserves_layout_and_reusable_viewer(self) -> None:
        workspace = AdaptiveImageWorkspace()
        viewer = ImageViewer()
        workspace.set_layout("4")
        workspace.set_results([FigureResult("Reusable", np.ones((2, 2)), viewer=viewer)])
        workspace.reset_button.click()
        QApplication.processEvents()
        self.assertEqual(workspace.results, [])
        self.assertEqual(workspace.current_page, 0)
        self.assertEqual(workspace.layout_choice.currentText(), "4")
        self.assertIs(viewer.parent(), workspace)
        workspace.update_result("reused", FigureResult("Reused", np.zeros((2, 2)), viewer=viewer))
        self.assertIs(workspace.panels[0].viewer, viewer)

    def test_figure_result_applies_scientific_colormap_and_linear_scaling(self) -> None:
        workspace = AdaptiveImageWorkspace()
        workspace.set_results([
            FigureResult("strain", np.asarray([[-1.0, 1.0]]), colormap="RdBu_r", scaling="linear")
        ])

        viewer = workspace.panels[0].viewer
        self.assertEqual(viewer.colormap, "RdBu_r")
        self.assertEqual(viewer.scaling, "linear")

    def test_rgb_result_uses_dedicated_viewer_and_display_only_flip(self) -> None:
        workspace = AdaptiveImageWorkspace()
        image = np.asarray([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        workspace.set_results([FigureResult("plot", image, image_kind="color", flip_x=True)])

        viewer = workspace.panels[0].viewer
        self.assertIsInstance(viewer, RgbImageViewer)
        np.testing.assert_array_equal(viewer.raw_image, image)
        self.assertEqual(viewer._pixmap.toImage().pixelColor(0, 0).blue(), 255)
        self.assertFalse(hasattr(viewer, "set_scaling"))


if __name__ == "__main__":
    unittest.main()
