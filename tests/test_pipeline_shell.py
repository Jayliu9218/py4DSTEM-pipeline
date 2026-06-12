import unittest
import types

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
)

from app.main_window import MainWindow
from app.services.phase_contrast_service import DPCStageResult, PhaseContrastResult
from app.services.workflow_state import WorkflowStep


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
            [
                "Data Setup",
                "BF / DF Preview",
                "Segmented DPC",
                "CoM Preprocessing",
                "CoM Review & Accept",
                "Integrated Reconstruction",
                "Export",
            ],
        )
        segmented = next(module for module in self.window.route_modules if module.key == "dpc_segmented")
        preprocess = next(module for module in self.window.route_modules if module.key == "dpc_preprocess")
        review = next(module for module in self.window.route_modules if module.key == "dpc_review")
        dpc = next(module for module in self.window.route_modules if module.key == "dpc")
        self.assertEqual(segmented.prerequisite, "data_setup")
        self.assertEqual(preprocess.prerequisite, "data_setup")
        self.assertEqual(review.prerequisite, "dpc_preprocess")
        self.assertEqual(dpc.prerequisite, "dpc_review")

    def test_dpc_stages_have_independent_workspaces(self) -> None:
        pages = [
            self.window.dpc_segmented_page,
            self.window.dpc_preprocess_page,
            self.window.dpc_review_page,
            self.window.dpc_reconstruction_page,
        ]

        self.assertEqual(len({id(page.workspace) for page in pages}), 4)
        self.assertEqual(len({id(page.service) for page in pages}), 1)

    def test_dpc_pages_complete_their_own_workflow_steps(self) -> None:
        segmented = self.window.dpc_segmented_page
        segmented.pending_operation = "Segmented DPC"
        segmented._handle_finished(
            DPCStageResult(
                stage="segmented",
                images={"Mean diffraction pattern": np.ones((2, 2))},
                masks=tuple(np.ones((2, 2)) for _ in range(4)),
            )
        )
        preprocess = self.window.dpc_preprocess_page
        preprocess.pending_operation = "DPC CoM preprocessing"
        preprocess._handle_finished(PhaseContrastResult(method="DPC"))

        self.assertTrue(self.window.workflow_state.is_completed(WorkflowStep.DPC_SEGMENTED))
        self.assertTrue(self.window.workflow_state.is_completed(WorkflowStep.DPC_PREPROCESS))

    def test_project_state_persists_staged_and_legacy_dpc_parameters(self) -> None:
        state = self.window._project_state()

        self.assertIn("dpc", state.page_params)
        self.assertIn("dpc_segmented", state.page_params)
        self.assertIn("dpc_preprocess", state.page_params)
        self.assertIn("dpc_review", state.page_params)
        self.assertIn("dpc_reconstruction", state.page_params)
        self.assertIn("dpc_legacy", state.page_params)
        self.assertIn("segment_outer_radius_mrad", state.page_params["dpc"])

    def test_com_review_controls_exist_only_in_review_stage(self) -> None:
        preprocess = self.window.dpc_preprocess_page
        review = self.window.dpc_review_page

        self.assertTrue(preprocess.accept_button.isHidden())
        self.assertTrue(preprocess.preprocess_view.isHidden())
        self.assertFalse(review.accept_button.isHidden())
        self.assertFalse(review.preprocess_view.isHidden())

    def test_preprocess_hands_result_to_review_without_duplicate_display(self) -> None:
        preprocess = self.window.dpc_preprocess_page
        review = self.window.dpc_review_page
        result = PhaseContrastResult(
            method="DPC",
            images={
                "Measured CoM X": np.ones((2, 2)),
                "Measured CoM Y": np.ones((2, 2)),
            },
        )
        preprocess.workspace.clear_results()
        preprocess.pending_operation = "DPC CoM preprocessing"

        preprocess._handle_finished(result)
        review.refresh_stage()

        self.assertEqual(preprocess.workspace.results, [])
        self.assertIs(review.result, result)
        self.assertGreater(len(review.workspace.results), 0)
        self.assertIn("Continue to CoM Review & Accept", preprocess.status_label.text())

    def test_parallax_goal_uses_focused_six_module_route(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        self.window.project_toolbar.goal.setCurrentText("Parallax")

        self.assertEqual(
            [module.title for module in self.window.route_modules],
            [
                "Data Setup", "BF Disk & Virtual BF", "Parallax Alignment",
                "Alignment Review", "Advanced Reconstruction", "Export",
            ],
        )
        alignment = next(
            module for module in self.window.route_modules if module.key == "parallax_alignment"
        )
        self.assertEqual(alignment.page_key, "parallax_alignment")
        self.assertIs(
            self.window._controls_for_route("parallax_alignment"),
            self.window.parallax_alignment_page.controls_panel,
        )
        self.assertTrue(self.window.parallax_advanced_page.subpixel.isChecked())
        self.assertFalse(self.window.parallax_advanced_page.aberration_fit.isChecked())
        self.assertFalse(self.window.parallax_advanced_page.aberration_correction.isChecked())
        self.assertFalse(self.window.parallax_advanced_page.high_order.isChecked())
        self.assertFalse(self.window.parallax_advanced_page.ctf_fit.isChecked())

    def test_route_module_selection_works(self) -> None:
        self.window._select_route_module("bragg_detection")
        self.assertEqual(self.window.current_route_key, "bragg_detection")

    def test_probe_and_strain_routes_preserve_shell_geometry(self) -> None:
        self.window.resize(1366, 768)
        self.window.show()
        self.app.processEvents()
        expected_log_height = self.window.log_panel.height()
        expected_splitter_sizes = self.window.main_splitter.sizes()
        for route in (
            "bragg_detection",
            "calibration",
            "crystal_analysis",
            "bragg_detection",
            "calibration",
            "crystal_analysis",
        ):
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

    def test_only_existing_calibration_form_wraps(self) -> None:
        page = self.window.calibration_page
        expected = {
            "Existing Calibration": QFormLayout.WrapAllRows,
            "Origin Calibration": QFormLayout.DontWrapRows,
            "Ellipse Calibration": QFormLayout.DontWrapRows,
            "Q Pixel Size": QFormLayout.DontWrapRows,
            "QR Rotation": QFormLayout.DontWrapRows,
            "Transfer": QFormLayout.DontWrapRows,
        }
        for title, policy in expected.items():
            form = page.calibration_forms[title]
            self.assertEqual(form.rowWrapPolicy(), policy)

    def test_calibration_uses_one_width_following_scroll_area(self) -> None:
        self.window.resize(1366, 768)
        self.window.show()
        self.window._select_route_module("calibration")
        self.app.processEvents()

        page = self.window.calibration_page
        self.assertNotIsInstance(page.controls_panel, QScrollArea)
        module_scrolls = self.window.module_panel.controls_stack.findChildren(QScrollArea)
        active_scroll = self.window.module_panel.controls_stack.currentWidget()
        self.assertIsInstance(active_scroll, QScrollArea)
        self.assertEqual(active_scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
        self.assertEqual(active_scroll.frameShape(), QFrame.NoFrame)
        self.assertEqual(len(page.controls_panel.findChildren(QScrollArea)), 0)
        self.assertLessEqual(page.controls_panel.minimumWidth(), active_scroll.viewport().width())
        self.assertEqual(active_scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(page.controls_panel.width(), active_scroll.viewport().width())

    def test_calibration_fields_expand_equally_within_each_group(self) -> None:
        self.window.resize(1366, 768)
        self.window.show()
        self.window._select_route_module("calibration")
        self.app.processEvents()

        page = self.window.calibration_page
        for controls in [
            [page.origin_center_x, page.origin_center_y, page.origin_robust_steps],
            [page.ellipse_center_x, page.ellipse_inner, page.sampling_spin],
            [page.rotation_spin, page.rotation_real_x, page.rotation_q_length],
        ]:
            widths = [control.width() for control in controls]
            self.assertLessEqual(max(widths) - min(widths), 1)

    def test_shell_boundaries_use_plain_black_lines(self) -> None:
        self.assertEqual(self.window.tree.frameShape(), QFrame.NoFrame)
        self.assertIn("border: 1px solid black", self.window.tree.styleSheet())
        self.assertIn("border-left: 1px solid black", self.window.module_panel.styleSheet())
        self.assertEqual(self.window.workflow_divider.height(), 1)
        self.assertIn("background: black", self.window.workflow_divider.styleSheet())
        self.assertEqual(self.window.log_divider.height(), self.window.workflow_divider.height())
        self.assertEqual(self.window.log_divider.width(), self.window.workflow_divider.width())
        self.assertEqual(self.window.log_divider.frameShape(), QFrame.NoFrame)
        self.assertEqual(self.window.workflow_divider.frameShape(), QFrame.NoFrame)

    def test_strain_reference_modes_are_notebook_aligned_and_migrate_legacy_values(self) -> None:
        page = self.window.strain_map_page
        self.assertEqual(
            [page.reference_mode.itemData(index) for index in range(page.reference_mode.count())],
            ["global_none", "roi_g1g2"],
        )
        self.assertFalse(hasattr(page, "manual_g1_x"))
        expected = {
            "auto_valid": "global_none",
            "roi_mask": "global_none",
            "manual_g1g2": "global_none",
            "roi_vectors": "roi_g1g2",
        }
        for legacy, migrated in expected.items():
            page.apply_params_snapshot({"reference_mode": legacy})
            self.assertEqual(page.reference_mode.currentData(), migrated)

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

    def test_calibration_origin_process_maps_use_signed_colormap(self) -> None:
        page = self.window.calibration_page
        page.viewers.clear_results()
        page._set_viewer_tab("qx residual", np.asarray([[-1.0, 1.0]]))

        result = page.viewers.results[-1]
        self.assertEqual(result.colormap, "RdBu_r")
        self.assertEqual(result.scaling, "linear")

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
