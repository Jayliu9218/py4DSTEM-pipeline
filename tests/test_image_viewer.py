import unittest

import numpy as np
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
