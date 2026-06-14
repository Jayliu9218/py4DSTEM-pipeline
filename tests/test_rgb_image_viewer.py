import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from app.widgets.rgb_image_viewer import RgbImageViewer


class RgbImageViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_click_maps_scaled_centered_pixmap_to_original_coordinates(self) -> None:
        viewer = RgbImageViewer()
        viewer.resize(400, 300)
        viewer.image_label.resize(400, 300)
        viewer.set_image(np.zeros((100, 200, 3), dtype=np.uint8))
        clicked = []
        viewer.image_clicked.connect(lambda x, y: clicked.append((x, y)))

        viewer._emit_image_click(200, 150)
        viewer._emit_image_click(200, 20)

        self.assertEqual(clicked, [(100, 50)])

    def test_flip_x_click_maps_back_to_original_coordinate(self) -> None:
        viewer = RgbImageViewer()
        viewer.resize(200, 100)
        viewer.image_label.resize(200, 100)
        viewer.set_image(np.zeros((100, 200, 3), dtype=np.uint8), flip_x=True)
        clicked = []
        viewer.image_clicked.connect(lambda x, y: clicked.append((x, y)))

        viewer._emit_image_click(0, 0)

        self.assertEqual(clicked, [(199, 0)])
