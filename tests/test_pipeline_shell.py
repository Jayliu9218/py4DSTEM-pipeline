import unittest
import types
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
)

from app.main_window import MainWindow
from app.services.bragg_strain_service import CalibrationActionResult
from app.services.phase_contrast_service import DPCStageResult, PhaseContrastResult
from app.services.workflow_state import WorkflowStep
from app.widgets.adaptive_image_workspace import FigureResult


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
            ["Data & Preprocess", "Virtual Imaging", "Probe & Bragg", "Calibration", "Strain Analysis", "Results & Quality", "Export"],
        )
        self.assertFalse(any(title.startswith("Step") for title in titles))
        strain = next(module for module in self.window.route_modules if module.key == "crystal_analysis")
        self.assertEqual(strain.prerequisite, "calibration")
        self.assertIn("calibration is recommended", strain.requirements)

    def test_orientation_and_results_use_canonical_shared_workspaces(self) -> None:
        self.assertIs(self.window.orientation_page, self.window.orientation_setup_page)
        self.assertIs(self.window.orientation_plan_page, self.window.orientation_setup_page)
        self.assertIs(self.window.orientation_review_page, self.window.orientation_setup_page)
        self.assertIs(
            self.window.orientation_map_page.workspace,
            self.window.crystalline_results_page.workspace,
        )
        workspaces = self.window.pages.named_workspaces()
        self.assertIn("orientation", workspaces)
        self.assertIn("crystalline_results", workspaces)
        self.assertNotIn("orientation_plan", workspaces)
        self.assertNotIn("orientation_review", workspaces)
        self.assertNotIn("orientation_map", workspaces)

    def test_legacy_orientation_grid_restores_into_shared_workspace(self) -> None:
        self.window.project_coordinator.restore_grid_states(
            {"orientation_review": {"layout": "4", "page": 3}}
        )
        self.assertEqual(self.window.orientation_setup_page.workspace.grid_state()["layout"], "4")
        self.assertEqual(self.window.crystalline_results_page.workspace.grid_state()["layout"], "4")

    def test_orientation_grid_layout_syncs_without_syncing_page(self) -> None:
        setup = self.window.orientation_setup_page.workspace
        results = self.window.crystalline_results_page.workspace
        setup.set_results([FigureResult(str(index), np.ones((2, 2))) for index in range(6)])
        results.set_results([
            FigureResult(str(index), np.ones((2, 2))) for index in range(6)
        ])
        results.set_layout("2")
        results.set_page(1)

        setup.set_layout("1")

        self.assertEqual(results.layout_choice.currentText(), "1")
        self.assertEqual(results.current_page, 1)

    def test_crystalline_export_is_final_and_keeps_results_workspace(self) -> None:
        self.assertEqual(self.window.route_modules[-1].key, "export")
        self.assertEqual(self.window.route_modules[-1].page_key, "crystalline_results")
        self.window._select_route_module("export")
        self.assertIs(
            self.window.viewer_stack.currentWidget(),
            self.window.crystalline_results_page,
        )

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
                "CoM Preprocessing & Review",
                "Integrated Reconstruction",
                "Export",
            ],
        )
        segmented = next(module for module in self.window.route_modules if module.key == "dpc_segmented")
        preprocess = next(module for module in self.window.route_modules if module.key == "dpc_preprocess")
        dpc = next(module for module in self.window.route_modules if module.key == "dpc")
        self.assertEqual(segmented.prerequisite, "data_setup")
        self.assertEqual(preprocess.prerequisite, "data_setup")
        self.assertEqual(dpc.prerequisite, "dpc_preprocess")

    def test_dpc_stages_have_independent_workspaces(self) -> None:
        pages = [
            self.window.dpc_segmented_page,
            self.window.dpc_preprocess_page,
            self.window.dpc_reconstruction_page,
        ]

        self.assertEqual(len({id(page.workspace) for page in pages}), 3)
        self.assertEqual(len({id(page.service) for page in pages}), 1)

    def test_dpc_pages_complete_their_own_workflow_steps(self) -> None:
        segmented = self.window.dpc_segmented_page
        segmented.pending_operation = "Segmented DPC"
        segmented._handle_result(
            DPCStageResult(
                stage="segmented",
                images={"Mean diffraction pattern": np.ones((2, 2))},
                masks=tuple(np.ones((2, 2)) for _ in range(4)),
            )
        )
        preprocess = self.window.dpc_preprocess_page
        preprocess.pending_operation = "DPC CoM preprocessing"
        preprocess._handle_result(PhaseContrastResult(method="DPC"))

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

    def test_com_review_controls_are_merged_into_preprocessing(self) -> None:
        preprocess = self.window.dpc_preprocess_page

        self.assertFalse(preprocess.accept_button.isHidden())
        self.assertFalse(preprocess.preprocess_view.isHidden())
        self.assertIs(self.window.dpc_review_page, preprocess)

    def test_preprocess_displays_review_and_requires_acceptance(self) -> None:
        preprocess = self.window.dpc_preprocess_page
        result = PhaseContrastResult(
            method="DPC",
            images={
                "Measured CoM X": np.ones((2, 2)),
                "Measured CoM Y": np.ones((2, 2)),
            },
        )
        preprocess.workspace.clear_results()
        preprocess.pending_operation = "DPC CoM preprocessing"

        preprocess._handle_result(result)

        self.assertIs(preprocess.result, result)
        self.assertGreater(len(preprocess.workspace.results), 0)
        self.assertIn("explicitly accept preprocessing", preprocess.status_label.text())

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
        self.assertTrue(self.window.parallax_advanced_page.subpixel_button.isEnabled() is False)
        self.assertFalse(self.window.parallax_advanced_page.high_order.isChecked())
        self.assertFalse(self.window.parallax_advanced_page.ctf_fit.isChecked())

    def test_parallax_alignment_presets_switch_between_fast_and_notebook_quality(self) -> None:
        page = self.window.parallax_alignment_page

        self.assertEqual(page.alignment_preset.currentText(), "Fast")
        self.assertEqual(page._alignment_params().alignment_bin_values, (32, 32, 16, 16, 8, 8))
        self.assertEqual(page._alignment_params().cross_correlation_upsample_factor, 4)

        page.alignment_preset.setCurrentText("Notebook Quality")

        self.assertEqual(
            page._alignment_params().alignment_bin_values,
            (32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8),
        )
        self.assertEqual(page._alignment_params().cross_correlation_upsample_factor, 8)

    def test_parallax_results_are_not_retained_for_method_comparison(self) -> None:
        self.assertFalse(hasattr(self.window, "_store_parallax_result"))
        self.assertNotIn("Parallax", self.window.phase_retrieval_results)

    def test_parallax_review_skips_unchanged_result_redraw(self) -> None:
        page = self.window.parallax_review_page
        result = PhaseContrastResult(
            method="Parallax",
            images={
                "Aligned BF": np.ones((4, 4)),
                "Shift Magnitude": np.ones((4, 4)),
            },
        )
        page.service.context.alignment_result = result
        page.service.context.shift_vectors = np.ones((16, 4))
        page._display_signature = None

        with patch.object(page.workspace, "set_results") as set_results:
            page._refresh_display()
            page._refresh_display()

        set_results.assert_called_once()
        figures = set_results.call_args.args[0]
        self.assertEqual([figure.title for figure in figures], ["Aligned BF", "Shift Magnitude"])
        self.assertEqual(figures[1].vector_stride, 4)

    def test_parallax_exposes_on_demand_notebook_diagnostic_views(self) -> None:
        review = self.window.parallax_review_page
        advanced = self.window.parallax_advanced_page

        self.assertIn("Notebook review", [review.review_view.itemText(i) for i in range(review.review_view.count())])
        self.assertIn("Finite-dose diffraction montage", [review.review_view.itemText(i) for i in range(review.review_view.count())])
        advanced_views = [advanced.advanced_view.itemText(i) for i in range(advanced.advanced_view.count())]
        self.assertIn("Original vs subpixel FFT", advanced_views)
        self.assertIn("Measured vs fitted shifts", advanced_views)
        self.assertIn("CTF comparison", advanced_views)

    def test_accept_alignment_emits_one_state_change_without_redraw(self) -> None:
        page = self.window.parallax_review_page
        page.service.context.alignment_result = PhaseContrastResult(
            method="Parallax", images={"Aligned BF": np.ones((2, 2))}
        )
        changes = []
        self.window.workflow_state.changed.connect(lambda: changes.append(True))

        with patch.object(page.workspace, "set_results") as set_results:
            page.accept_alignment()

        set_results.assert_not_called()
        self.assertEqual(len(changes), 1)

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

    def test_calibration_empty_apply_result_preserves_current_images(self) -> None:
        page = self.window.calibration_page
        page.viewers.clear_results()
        page._set_viewer_tab("qx residual", np.asarray([[-1.0, 1.0]]))
        before = list(page.viewers.results)
        page.current_process_name = "Apply origin correction"

        page._handle_result(CalibrationActionResult("Applied corrections: center.", {}, 0.01))

        self.assertEqual(page.viewers.results, before)

    def test_module_panel_gives_top_level_groups_comfortable_spacing(self) -> None:
        self.window._select_route_module("crystal_analysis")
        controls = self.window.strain_map_page.controls_panel
        self.assertEqual(controls.layout().spacing(), 12)
        for index in range(controls.layout().count()):
            widget = controls.layout().itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                self.assertEqual(widget.sizePolicy().verticalPolicy(), QSizePolicy.Fixed)
                self.assertIn("border-radius: 6px", widget.styleSheet())
                self.assertEqual(widget.minimumHeight(), widget.maximumHeight())

    def test_data_setup_does_not_double_shared_workspace_margin(self) -> None:
        margins = self.window.main_view.layout().contentsMargins()
        self.assertEqual(
            (margins.left(), margins.top(), margins.right(), margins.bottom()),
            (0, 0, 0, 0),
        )

    def test_all_workspace_pages_use_data_setup_outer_alignment(self) -> None:
        for key, workspace in self.window.pages.named_workspaces().items():
            page = self.window.viewer_pages[key]
            margins = page.layout().contentsMargins()
            self.assertEqual(
                (margins.left(), margins.top(), margins.right(), margins.bottom()),
                (0, 0, 0, 0),
                key,
            )
            self.assertEqual(workspace.layout_choice.width(), 80, key)

    def test_existing_parameter_tables_use_fixed_scrollable_style(self) -> None:
        self.window.project_toolbar.goal.setCurrentText("Orientation Mapping")
        self.window._select_route_module("orientation_setup")
        table = self.window.orientation_setup_page.atom_table
        self.assertEqual(table.height(), 180)
        self.assertIn("font-weight: bold", table.styleSheet())
        self.assertEqual(table.verticalScrollBarPolicy(), Qt.ScrollBarAsNeeded)

    def test_orientation_rgb_map_click_returns_to_single_pattern_review(self) -> None:
        target = self.window.orientation_setup_page
        mapping = self.window.orientation_map_page
        mapping.workspace.set_results([
            FigureResult("Orientation RGB", np.zeros((3, 4, 3)), image_kind="rgb")
        ])

        mapping._connect_map_clicks()
        mapping.workspace.panels[0].viewer.image_clicked.emit(2, 1)

        self.assertEqual(target.scan_x.value(), 2)
        self.assertEqual(target.scan_y.value(), 1)
        self.assertIn("Selected map position", target.status_label.text())

    def test_full_orientation_map_locks_auto_grid_before_calculation(self) -> None:
        setup = self.window.orientation_setup_page.workspace
        mapping_page = self.window.orientation_map_page
        mapping_page.workspace.set_results([
            FigureResult(str(index), np.ones((2, 2))) for index in range(4)
        ])
        mapping_page.workspace.set_layout("Auto")

        with patch("app.widgets.worker_runner.QThread.start"):
            mapping_page._start("Full Orientation Map", lambda: None)

        locked = mapping_page.workspace.layout_choice.currentText()
        self.assertNotEqual(locked, "Auto")
        self.assertEqual(setup.layout_choice.currentText(), locked)
        mapping_page._clear_worker_refs()

    def test_parameter_inputs_remain_left_aligned(self) -> None:
        self.assertEqual(
            self.window.orientation_setup_page.voltage.line_edit.alignment(),
            Qt.AlignLeft,
        )

    def test_orientation_grid_pages_have_no_top_status_bar(self) -> None:
        setup = self.window.orientation_setup_page
        results = self.window.crystalline_results_page
        self.assertIs(setup.layout().itemAt(0).widget(), setup.workspace)
        self.assertIs(results.layout().itemAt(0).widget(), results.workspace)
        self.assertFalse(setup.status_label.isVisible())

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
