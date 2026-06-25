import sys
import types
import unittest

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

sys.modules.setdefault("py4DSTEM", types.SimpleNamespace())

from main import _load_stylesheet
from app.main_window import MainWindow
from app.qt_utils import WheelEventFilter
from app.version import __version__
from app.services.project_state_service import ProjectState
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
        self.kwargs_seen = {}

    def get_dp_mean(self):
        return types.SimpleNamespace(data=np.ones((2, 2)))

    def find_Bragg_disks(self, **kwargs):
        self.cuda_seen = kwargs.get("CUDA")
        self.kwargs_seen = kwargs
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

        self.assertEqual(top_level[:6], ["Files", "Mode", "Layout", "View", "Setting", "Help"])
        self.assertIn("Setting", top_level)
        self.assertNotIn("Setting", file_actions)
        self.assertFalse(hasattr(window, "setting_menu"))

    def test_industrial_light_theme_is_default_and_dark_remains_available(self) -> None:
        window = MainWindow()

        self.assertTrue(window.light_theme_action.isChecked())
        self.assertFalse(window.dark_theme_action.isChecked())
        self.assertIn("Compact industrial-light instrument theme", _load_stylesheet())

        window._apply_theme("dark")
        self.assertIn("SEM/FIB dark-gray instrument theme", self.app.styleSheet())
        window._apply_theme("light")
        self.assertIn("Compact industrial-light instrument theme", self.app.styleSheet())

    def test_combo_wheel_changes_are_blocked_without_affecting_other_widgets(self) -> None:
        event_filter = WheelEventFilter()
        wheel_event = QEvent(QEvent.Wheel)

        self.assertTrue(event_filter.eventFilter(QComboBox(), wheel_event))
        self.assertFalse(event_filter.eventFilter(QLabel(), wheel_event))

    def test_help_menu_actions_are_descriptive_and_have_complete_content(self) -> None:
        window = MainWindow()
        actions = [action.text().replace("&", "") for action in window.help_menu.actions()]

        self.assertEqual(actions, ["About", "License", "Workflow Tutorials"])
        self.assertIn("current capabilities", window.about_action.statusTip())
        self.assertIn("GNU GPLv3", window.license_action.statusTip())
        self.assertIn("each analysis workflow", window.tutorials_action.statusTip())
        self.assertIn("Current situation", window.ABOUT_HTML)
        self.assertIn("Current improvements", window.ABOUT_HTML)
        self.assertIn(__version__, window.ABOUT_HTML)
        self.assertIn("GNU General Public License version 3", window.LICENSE_HTML)
        self.assertIn("Show Data", window.TUTORIAL_HTML)
        self.assertIn("Apply Best Self-Consistency Value", window.TUTORIAL_HTML)
        self.assertIn("accepted QC remains valid", window.TUTORIAL_HTML)
        self.assertIn("cancellable background tasks", window.TUTORIAL_HTML)
        for workflow in (
            "Crystal Analysis",
            "DPC / CoM", "Parallax", "Ptychography", "Method Comparison",
            "RDF", "FEM", "Amorphous Strain",
        ):
            self.assertIn(workflow, window.TUTORIAL_HTML)

    def test_cuda_setting_updates_pages_without_file_state(self) -> None:
        window = MainWindow()

        window._apply_cuda_setting(True)

        self.assertTrue(window.cuda_enabled)
        self.assertTrue(window.bragg_peaks_page.cuda_enabled)
        self.assertTrue(window.orientation_page.cuda_enabled)
        self.assertEqual(window.parallax_alignment_page._alignment_params().device, "gpu")

    def test_project_state_restores_settings_roles_and_page_params(self) -> None:
        window = MainWindow()
        state = ProjectState(
            image_scaling="linear",
            image_cmap="magma",
            cuda_enabled=True,
            dataset_roles={"target_datacube": "/data"},
            page_params={
                "virtual_detector": {"mode": "Annular Dark Field", "center_x": 22},
                "calibration": {"analysis_target": "Strain", "q_pixel_size": 0.04},
            },
        )

        window._apply_project_state(state)

        self.assertEqual(window.image_scaling, "linear")
        self.assertEqual(window.image_cmap, "magma")
        self.assertTrue(window.cuda_enabled)
        self.assertEqual(window.workflow_state.dataset_roles.target_datacube, "/data")
        self.assertEqual(window.virtual_detector_page.center_x_spin.value(), 22)
        self.assertEqual(window.calibration_page.analysis_target.currentText(), "Strain")
        self.assertEqual(window.calibration_page.pixel_spin.value(), 0.04)

    def test_braggvectors_forwards_cuda_flag(self) -> None:
        datacube = _DataCube()

        BraggStrainService().compute_braggvectors(
            datacube, BraggDetectionParams(cuda=True, allow_gaussian_fallback=True)
        )

        self.assertTrue(datacube.cuda_seen)

    def test_braggvectors_forwards_advanced_cbs_parameters(self) -> None:
        datacube = _DataCube()
        params = BraggDetectionParams(
            allow_gaussian_fallback=True,
            corr_power=0.75,
            sigma_dp=1.5,
            sigma_cc=0.5,
            upsample_factor=8,
            radial_background_subtraction=True,
        )

        BraggStrainService().compute_braggvectors(datacube, params)

        self.assertEqual(datacube.kwargs_seen["corrPower"], 0.75)
        self.assertEqual(datacube.kwargs_seen["sigma_dp"], 1.5)
        self.assertEqual(datacube.kwargs_seen["sigma_cc"], 0.5)
        self.assertEqual(datacube.kwargs_seen["upsample_factor"], 8)
        self.assertTrue(datacube.kwargs_seen["radial_bksb"])

    def test_orientation_plan_forwards_cuda_flag(self) -> None:
        crystal = _Crystal()
        service = OrientationService()
        service.crystal = crystal

        service.create_plan(OrientationPlanParams(cuda=True))

        self.assertTrue(crystal.cuda_seen)


if __name__ == "__main__":
    unittest.main()
