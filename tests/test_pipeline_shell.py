import unittest
import types

import numpy as np
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
        strain = next(module for module in self.window.route_modules if module.key == "crystal_analysis")
        self.assertEqual(strain.prerequisite, "bragg_detection")
        self.assertIn("calibration is recommended", strain.requirements)

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

    def test_probe_and_strain_routes_preserve_shell_geometry(self) -> None:
        self.window.resize(1366, 768)
        self.window.show()
        self.app.processEvents()
        expected_log_height = self.window.log_panel.height()
        expected_splitter_sizes = self.window.main_splitter.sizes()
        for route in ("bragg_detection", "crystal_analysis", "bragg_detection", "crystal_analysis"):
            self.window._select_route_module(route)
            self.app.processEvents()
            self.assertEqual(self.window.log_panel.height(), expected_log_height)
            self.assertEqual(self.window.main_splitter.sizes(), expected_splitter_sizes)
        self.assertEqual(expected_log_height, 140)

    def test_console_is_compact_without_redundant_labels(self) -> None:
        labels = [label.text() for label in self.window.log_panel.findChildren(type(self.window.path_label))]
        self.assertNotIn("Console", labels)
        self.assertNotIn("Calculation process", labels)
        self.assertEqual(self.window.log_panel.height(), 140)

    def test_calibration_has_visible_reset_action(self) -> None:
        self.assertEqual(
            self.window.calibration_page.reset_button.text(),
            "Reset Applied Calibration",
        )

    def test_existing_calibration_shows_values_and_applied_state(self) -> None:
        calibration = types.SimpleNamespace(
            get_origin=lambda: (4, 5),
            get_ellipse=lambda: (3, 2, 0.1),
            get_Q_pixel_size=lambda: 0.02,
            get_Q_pixel_units=lambda: "A^-1",
            get_QR_rotation_degrees=lambda: -83,
        )
        self.window.bragg_strain_service.braggvectors = types.SimpleNamespace(
            calibration=calibration,
            calstate={"center": True, "ellipse": False, "pixel": True, "rotate": True},
            histogram=lambda mode="raw", sampling=1: types.SimpleNamespace(data=np.ones((3, 3))),
        )

        page = self.window.calibration_page
        page.analysis_target.setCurrentText("Strain")
        page.refresh_status()

        self.assertEqual(page.origin_label.text(), "x=4, y=5 [applied]")
        self.assertIn("ellipticity=1.5 [not applied]", page.ellipse_label.text())
        self.assertEqual(page.pixel_label.text(), "0.02 A^-1 [applied]")
        self.assertEqual(page.rotate_label.text(), "-83 deg [applied]")
        self.assertTrue(all(label.text() == "Recommended" for label in page.decision_labels.values()))

    def test_calibration_bragg_sampling_callback_has_numpy_available(self) -> None:
        class Histogram:
            data = np.ones((3, 3))

        braggvectors = types.SimpleNamespace(
            histogram=lambda mode="raw", sampling=1: Histogram(),
            calibration=types.SimpleNamespace(),
            calstate={},
        )
        self.window.bragg_strain_service.braggvectors = braggvectors
        self.window.calibration_page.show_braggvectors_histogram()

        provider = self.window.calibration_page.viewers.results[-1].bragg_sampling_provider

        self.assertIsNotNone(provider)
        np.testing.assert_array_equal(provider(2), np.ones((3, 3)))

    def test_full_bragg_result_displays_only_full_map_from_handler(self) -> None:
        page = self.window.bragg_peaks_page
        page.workspace.clear_results()
        quality = types.SimpleNamespace(
            peak_count_map=np.ones((2, 2)),
            mean_intensity_map=np.ones((2, 2)),
            max_intensity_map=np.ones((2, 2)),
            failure_mask=np.zeros((2, 2), dtype=bool),
        )
        page._handle_braggvectors_result(types.SimpleNamespace(
            peak_count=4,
            elapsed_seconds=0.1,
            bragg_vector_map=np.ones((4, 4)),
            quality=quality,
        ))

        self.assertEqual([result.title for result in page.workspace.results], ["Full Bragg Vector Map"])

    def test_strain_display_uses_py4dstem_colormaps_and_keeps_coordinate_rotation(self) -> None:
        page = self.window.strain_map_page
        original_rotation = page.rotation_spin.value()
        page.workspace.clear_results()
        page.result = types.SimpleNamespace(
            components={
                "exx": np.asarray([[-1.0, 1.0]]),
                "eyy": np.asarray([[-1.0, 1.0]]),
                "exy": np.asarray([[-1.0, 1.0]]),
                "theta": np.asarray([[-1.0, 1.0]]),
            }
        )
        page._display_result()

        self.assertEqual(
            [result.colormap for result in page.workspace.results],
            ["RdBu_r", "RdBu_r", "RdBu_r", "PRGn"],
        )
        self.assertTrue(all(result.scaling == "linear" for result in page.workspace.results))
        self.assertEqual(page.rotation_spin.value(), original_rotation)

    def test_strain_process_and_final_views_replace_the_grid(self) -> None:
        page = self.window.strain_map_page
        page.result = types.SimpleNamespace(
            components={"exx": np.ones((2, 2)), "theta": np.ones((2, 2))},
            process_images={"basis selection": np.ones((3, 3)), "reference mask": np.ones((2, 2))},
            process_vectors={"basis selection": np.asarray([[1, 1, 1, 0]])},
        )
        page.display_mode.setCurrentText("Process")
        page._display_result()
        self.assertEqual([result.title for result in page.workspace.results], ["basis selection", "reference mask"])
        self.assertIsNotNone(page.workspace.results[0].vectors)
        page.display_mode.setCurrentText("Final Strain")
        self.assertEqual([result.title for result in page.workspace.results], ["exx", "theta"])

    def test_calibration_values_wrap_and_ellipse_requires_acceptance(self) -> None:
        page = self.window.calibration_page
        self.assertTrue(page.ellipse_label.wordWrap())
        self.assertTrue(page.ellipse_measurement_label.wordWrap())
        self.assertEqual(page.apply_ellipse_button.text(), "Accept && Apply Ellipse")

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
