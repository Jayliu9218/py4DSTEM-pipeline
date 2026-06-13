import unittest

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from app.widgets.image_viewer import ImageViewer


class ImageViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_nan_only_image_clears_with_message(self) -> None:
        viewer = ImageViewer()

        viewer.set_image(np.asarray([[np.nan, np.nan]]))

        self.assertEqual(viewer.message_item.toPlainText(), "Image contains no finite values.")

    def test_constant_image_gets_nonzero_levels(self) -> None:
        viewer = ImageViewer()

        levels = viewer._safe_levels(np.ones((2, 2)), None)

        self.assertLess(levels[0], levels[1])

    def test_default_scaling_is_log_and_can_switch_linear(self) -> None:
        viewer = ImageViewer()

        self.assertEqual(viewer.scaling, "log")
        viewer.set_scaling("linear")

        self.assertEqual(viewer.scaling, "linear")

    def test_colormap_can_be_changed(self) -> None:
        viewer = ImageViewer()

        viewer.set_colormap("magma")

        self.assertEqual(viewer.colormap, "magma")

    def test_bragg_sampling_provider_can_redraw_image(self) -> None:
        viewer = ImageViewer()
        viewer.set_bragg_sampling_provider(lambda sampling: np.full((2, 2), sampling))

        viewer.bragg_sampling = 3
        viewer.set_image(viewer.bragg_sampling_provider(viewer.bragg_sampling))

        np.testing.assert_array_equal(viewer.raw_image, np.full((2, 2), 3))

    def test_context_menu_has_display_options_and_optional_sampling(self) -> None:
        viewer = ImageViewer()
        base_actions = [action.text() for action in viewer._create_context_menu().actions()]
        viewer.set_bragg_sampling_provider(lambda sampling: np.full((2, 2), sampling))
        bragg_actions = [action.text() for action in viewer._create_context_menu().actions()]

        self.assertIn("Scaling", base_actions)
        self.assertIn("Colormap", base_actions)
        self.assertTrue(any(action.startswith("BraggVectors Sampling") for action in bragg_actions))

    def test_interactive_roi_rect_can_be_set_and_read(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((10, 12)))

        viewer.set_interactive_roi_rect(2, 6, 3, 8)

        self.assertEqual(viewer.interactive_roi_rect(), (2, 6, 3, 8))
        self.assertEqual(tuple(viewer.interactive_roi_item.pos()), (3.0, 2.0))
        self.assertEqual(tuple(viewer.interactive_roi_item.size()), (5.0, 4.0))

    def test_interactive_circle_can_be_set_and_read(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((10, 12)))

        viewer.set_interactive_circle(5, 6, 3)

        self.assertEqual(viewer.interactive_circle(), (5.0, 6.0, 3.0))

    def test_interactive_annulus_has_independent_radii_and_shared_center(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((20, 20)))

        viewer.set_interactive_annulus(10, 11, 3, 7)

        self.assertEqual(viewer.interactive_annulus(), (10.0, 11.0, 3.0, 7.0))

    def test_interactive_ellipse_can_be_set_and_read(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((10, 12)))

        viewer.set_interactive_ellipse(5, 6, 3, 2, 15)

        self.assertEqual(viewer.interactive_ellipse(), (5.0, 6.0, 3.0, 2.0, 15.0))

    def test_vector_overlays_can_be_added_and_cleared(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((8, 8)))
        viewer.add_vector_overlays(np.asarray([[2, 3, 1, -1], [4, 5, -2, 1]]))
        self.assertEqual(len(viewer.overlay_items), 1)
        first_x, first_y = viewer.overlay_items[0].getData()
        np.testing.assert_array_equal(first_x, [3, 2, np.nan, 5, 6, np.nan])
        np.testing.assert_array_equal(first_y, [2, 3, np.nan, 4, 2, np.nan])
        viewer.clear_overlays()
        self.assertEqual(viewer.overlay_items, [])

    def test_notebook_semantic_labels_and_mask_overlay_align_with_scalar_image(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((3, 5)))
        viewer.add_point_labels(np.asarray([[1, 4]]), ["g1"])
        viewer.add_mask_overlay(np.asarray([
            [False, False, False, False, False],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ]))

        self.assertEqual(len(viewer.overlay_items), 2)
        position = viewer.overlay_items[0].pos()
        self.assertEqual((position.x(), position.y()), (4.0, 1.0))
        self.assertEqual(viewer.overlay_items[1].axisOrder, "row-major")

    def test_scalar_image_uses_matplotlib_row_major_orientation_without_transposing_data(self) -> None:
        viewer = ImageViewer()
        image = np.arange(15, dtype=float).reshape(3, 5)

        viewer.set_image(image)

        self.assertEqual(viewer.image_view.getImageItem().axisOrder, "row-major")
        np.testing.assert_array_equal(viewer.raw_image, image)
        np.testing.assert_array_equal(viewer.rendered_image, np.log1p(image))
        bounds = viewer.image_view.getImageItem().boundingRect()
        self.assertEqual((bounds.width(), bounds.height()), (5.0, 3.0))

    def test_scientific_points_and_geometry_are_swapped_only_for_display(self) -> None:
        viewer = ImageViewer()
        viewer.set_image(np.ones((10, 14)))

        viewer.set_points(np.asarray([2]), np.asarray([9]))
        point_x, point_y = viewer.scatter_item.getData()
        np.testing.assert_array_equal(point_x, [9])
        np.testing.assert_array_equal(point_y, [2])

        viewer.set_circle_overlay(3, 8, 2)
        self.assertEqual(tuple(viewer.overlay_items[0].pos()), (6.0, 1.0))

        viewer.set_ellipse_overlay(3, 8, 4, 2, np.pi / 6)
        ellipse = viewer.overlay_items[0]
        self.assertEqual(tuple(ellipse.pos()), (6.0, -1.0))
        self.assertEqual(tuple(ellipse.size()), (4.0, 8.0))
        self.assertAlmostEqual(float(ellipse.angle()), -30.0)

    def test_mouse_readout_uses_scientific_array_coordinates(self) -> None:
        viewer = ImageViewer()
        image = np.arange(15, dtype=float).reshape(3, 5)
        viewer.set_image(image)
        scene_pos = viewer.image_view.getImageItem().mapToScene(QPointF(4, 1))

        viewer._handle_mouse_moved(scene_pos)

        self.assertEqual(viewer.coordinate_label.text(), "x: 1, y: 4, value: 9")

    def test_intensity_scaling_changes_grayscale_display(self) -> None:
        viewer = ImageViewer()
        image = np.asarray([[0.0, 3.0]])

        viewer.set_image(image)
        log_display = viewer.rendered_image.copy()
        viewer.set_scaling("linear")

        self.assertFalse(np.allclose(log_display, viewer.rendered_image))
        np.testing.assert_allclose(viewer.rendered_image, image)

    def test_color_image_is_accepted_and_not_scaled(self) -> None:
        viewer = ImageViewer("color")
        image = np.asarray([[[0.0, 3.0, 8.0], [1.0, 4.0, 9.0]]])

        viewer.set_image(image)
        log_display = viewer.rendered_image.copy()
        viewer.set_scaling("linear")

        np.testing.assert_allclose(log_display, image)
        np.testing.assert_allclose(viewer.rendered_image, image)

    def test_invalid_dimensions_fail_for_viewer_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Intensity image viewer expects a 2D array"):
            ImageViewer().set_image(np.zeros((2, 2, 3)))

        with self.assertRaisesRegex(ValueError, "Color image viewer expects an RGB/RGBA array"):
            ImageViewer("color").set_image(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
