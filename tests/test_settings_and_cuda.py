import sys
import types
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

sys.modules.setdefault("py4DSTEM", types.SimpleNamespace())

from app.main_window import MainWindow
from app.services.bragg_strain_service import BraggDetectionParams, BraggStrainService
from app.services.orientation_service import OrientationPlanParams, OrientationService


class _Histogram:
    data = np.ones((2, 2))


class _BraggVectors:
    shape = (1, 1)
    raw = None

    def histogram(self, mode="raw"):
        return _Histogram()


class _DataCube:
    shape = (1, 1, 2, 2)
    data = np.ones(shape)

    def __init__(self):
        self.cuda_seen = None

    def get_dp_mean(self):
        return types.SimpleNamespace(data=np.ones((2, 2)))

    def find_Bragg_disks(self, **kwargs):
        self.cuda_seen = kwargs.get("CUDA")
        return _BraggVectors()


class _Crystal:
    def __init__(self):
        self.cuda_seen = None

    def setup_diffraction(self, **_kwargs):
        return None

    def calculate_structure_factors(self, **_kwargs):
        return None

    def orientation_plan(self, **kwargs):
        self.cuda_seen = kwargs.get("CUDA")


class SettingsAndCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_setting_is_top_level_not_file_menu_item(self) -> None:
        window = MainWindow()

        top_level = [action.text().replace("&", "") for action in window.menuBar().actions()]
        file_actions = [action.text().replace("&", "") for action in window.file_menu.actions()]

        self.assertIn("Setting", top_level)
        self.assertNotIn("Setting", file_actions)

    def test_cuda_setting_updates_pages_without_file_state(self) -> None:
        window = MainWindow()

        window._apply_cuda_setting(True)

        self.assertTrue(window.cuda_enabled)
        self.assertTrue(window.bragg_peaks_page.cuda_enabled)
        self.assertTrue(window.orientation_page.cuda_enabled)

    def test_braggvectors_forwards_cuda_flag(self) -> None:
        datacube = _DataCube()

        BraggStrainService().compute_braggvectors(datacube, BraggDetectionParams(cuda=True))

        self.assertTrue(datacube.cuda_seen)

    def test_orientation_plan_forwards_cuda_flag(self) -> None:
        crystal = _Crystal()
        service = OrientationService()
        service.crystal = crystal

        service.create_plan(OrientationPlanParams(cuda=True))

        self.assertTrue(crystal.cuda_seen)


if __name__ == "__main__":
    unittest.main()
