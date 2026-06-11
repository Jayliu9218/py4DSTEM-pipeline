import unittest

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QGroupBox, QPushButton, QTabWidget

from app.main_window import MainWindow


class PipelineShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.window.close()

    def setUp(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Crystalline / Bragg-based")
        self.window.project_toolbar.goal.setCurrentText("Strain Mapping")
        self.window.current_route_key = "data_setup"
        self.window._refresh_pipeline_state()

    def test_crystalline_route_uses_named_modules_not_numbered_steps(self) -> None:
        titles = [module.title for module in self.window.route_modules]

        self.assertEqual(
            titles,
            ["Data & Preprocess", "Virtual Imaging", "Probe & Bragg", "Calibration", "Strain Analysis", "Export & Report"],
        )
        self.assertFalse(any(title.startswith("Step") for title in titles))

    def test_structure_and_goal_change_rebuilds_route(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Amorphous / Diffuse-scattering")
        self.window.project_toolbar.goal.setCurrentText("FEM")

        titles = [module.title for module in self.window.route_modules]
        self.assertEqual(
            titles,
            ["Data Setup", "Radial Profile", "FEM", "Export"],
        )

    def test_phase_retrieval_route_builds_correctly(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        self.window.project_toolbar.goal.setCurrentText("DPC / CoM")

        titles = [module.title for module in self.window.route_modules]
        self.assertEqual(
            titles,
            ["Data Setup", "BF / DF Preview", "DPC / CoM", "Parallax", "Ptychography", "Method Comparison", "Export"],
        )

    def test_route_module_selection_works(self) -> None:
        self.window._select_route_module("bragg_detection")
        self.assertEqual(self.window.current_route_key, "bragg_detection")

    def test_calibration_has_visible_reset_action(self) -> None:
        self.assertEqual(
            self.window.calibration_page.reset_button.text(),
            "Reset Applied Calibration",
        )

    def test_main_window_uses_native_qt_style(self) -> None:
        self.assertEqual(self.window.styleSheet(), "")

    def test_only_group_box_titles_are_bold(self) -> None:
        self.assertTrue(all(group.font().bold() for group in self.window.findChildren(QGroupBox)))
        self.assertTrue(
            all(not tabs.tabBar().font().bold() for tabs in self.window.findChildren(QTabWidget))
        )
        self.assertTrue(
            all(
                not button.font().bold()
                for button in self.window.data_setup_controls.findChildren(QPushButton)
            )
        )


if __name__ == "__main__":
    unittest.main()
