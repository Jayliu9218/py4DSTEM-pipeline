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
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
)

from app.main_window import MainWindow
from app.theme import (
    ACTION_BUTTON_MIN_HEIGHT,
    GROUP_SPACING,
    PANEL_MARGIN,
    PANEL_MARGIN_TIGHT,
    PARAM_TABLE_HEIGHT,
)
from app.services.bragg_strain_service import CalibrationActionResult
from app.services.phase_contrast_service import DPCStageResult, PhaseContrastResult
from app.services.workflow_state import WorkflowStep
from app.widgets.adaptive_image_workspace import FigureResult
from app.widgets.scientific_controls import (
    SCI_CONTROLS_PANEL,
    SCI_PROPERTY_LABEL,
    SCI_SECTION,
)


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
        self.window.project_toolbar.goal.setCurrentText("Crystal Analysis")
        self.window.current_route_key = "data_setup"
        self.window._refresh_pipeline_state()

    def test_crystalline_route_uses_named_modules_not_numbered_steps(self) -> None:
        titles = [module.title for module in self.window.route_modules]

        self.assertEqual(
            titles,
            [
                "Data, Preprocess & Virtual Image",
                "Probe & Bragg",
                "Calibration",
                "Crystal Setup & Phase Matching",
                "Orientation & Grain",
                "Strain & Results",
                "Export",
            ],
        )
        self.assertFalse(any(title.startswith("Step") for title in titles))
        strain = next(module for module in self.window.route_modules if module.key == "strain_analysis")
        self.assertEqual(strain.prerequisite, "orientation_matching")
        self.assertIn("Phase masks", strain.requirements)

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
        self.assertEqual(self.window.route_modules[-1].page_key, "crystal_strain")
        self.window._select_route_module("export")
        self.assertIs(
            self.window.viewer_stack.currentWidget(),
            self.window.crystal_strain_page,
        )

    def test_phase_retrieval_export_routes_use_dedicated_pages(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        for goal, page in (
            ("Parallax", self.window.parallax_export_page),
            ("Ptychography", self.window.ptychography_export_page),
        ):
            self.window.project_toolbar.goal.setCurrentText(goal)
            export = self.window.route_modules[-1]
            self.assertEqual(export.page_key, f"{goal.lower()}_export")
            self.window._select_route_module("export")
            self.assertIs(self.window.viewer_stack.currentWidget(), page)

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

    def test_bf_df_preview_reuses_cached_display_results(self) -> None:
        page = self.window.bf_df_preview_page
        source = object()
        result = {"Bright Field": np.ones((2, 2)), "Dark Field": np.ones((2, 2)) * 2}
        page.source_provider = lambda: source
        page.service.compute_bf_df_task = types.MethodType(
            lambda _service, *_args, **_kwargs: self.fail("cached BF/DF result should not start a task"),
            page.service,
        )
        key = page._cache_key(source)
        page.result_cache.put(key, result)

        page._run()

        self.assertEqual(page.status_label.text(), "Done (cached)")
        self.assertIs(page.result, result)

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
        # With docks, the viewer stack size should stay stable across route switches.
        expected_viewer_height = self.window.viewer_stack.height()
        for route in (
            "bragg_detection",
            "calibration",
            "phase_setup",
            "bragg_detection",
            "calibration",
            "strain_analysis",
        ):
            self.window._select_route_module(route)
            self.app.processEvents()
            self.assertEqual(self.window.viewer_stack.height(), expected_viewer_height)

    def test_initial_dock_layout_matches_reset_layout(self) -> None:
        window = MainWindow()
        window.resize(1600, 900)
        window.show()
        self.app.processEvents()
        initial = tuple(
            (dock.geometry().x(), dock.geometry().y(), dock.width(), dock.height())
            for dock in (window.data_dock, window.controls_dock, window.output_dock)
        )

        window._reset_layout(silent=True)
        self.app.processEvents()
        reset = tuple(
            (dock.geometry().x(), dock.geometry().y(), dock.width(), dock.height())
            for dock in (window.data_dock, window.controls_dock, window.output_dock)
        )

        self.assertEqual(initial, reset)
        window.close()

    def test_console_is_compact_without_redundant_labels(self) -> None:
        labels = [label.text() for label in self.window.log_panel.findChildren(type(self.window.path_label))]
        self.assertNotIn("Console", labels)
        self.assertNotIn("Calculation process", labels)

    def test_calibration_has_visible_reset_action(self) -> None:
        self.assertEqual(
            self.window.calibration_page.reset_button.text(),
            "Reset Applied Calibration",
        )

    def test_calibration_uses_scientific_sections(self) -> None:
        page = self.window.calibration_page
        expected = [
            "Existing Calibration",
            "Origin Calibration",
            "Ellipse Calibration",
            "Q Pixel Size",
            "QR Rotation",
            "Transfer",
        ]
        self.assertEqual(page.controls_panel.objectName(), SCI_CONTROLS_PANEL)
        self.assertEqual(
            [page.calibration_forms[title].objectName() for title in expected],
            [SCI_SECTION] * len(expected),
        )

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

    def test_shell_boundaries_use_themed_dividers(self) -> None:
        # Dividers are now styled globally via theme.qss by objectName,
        # not via inline QSS. Verify the structural hooks are in place.
        self.assertEqual(self.window.tree.frameShape(), QFrame.NoFrame)
        self.assertEqual(self.window.module_panel.objectName(), "moduleControlPanel")
        self.assertEqual(self.window.workflow_divider.height(), 1)
        self.assertEqual(self.window.workflow_divider.objectName(), "workflowDivider")
        self.assertEqual(self.window.log_divider.height(), self.window.workflow_divider.height())
        self.assertEqual(self.window.log_divider.objectName(), "logDivider")
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
        self.window._select_route_module("strain_analysis")
        controls = self.window.crystal_strain_page.controls_panel
        self.assertEqual(controls.layout().spacing(), GROUP_SPACING)
        for index in range(controls.layout().count()):
            widget = controls.layout().itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                self.assertEqual(widget.sizePolicy().verticalPolicy(), QSizePolicy.Preferred)
                self.assertLess(widget.minimumHeight(), widget.maximumHeight())

    def test_dynamic_parameter_group_grows_for_multiline_warnings(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        self.window.project_toolbar.goal.setCurrentText("Ptychography")
        self.window._select_route_module("ptychography_data")
        page = self.window.ptychography_data_page
        group = page.groups["data"]
        before = group.sizeHint().height()

        page.suitability_label.setText("\n".join(["Long suitability warning " * 8] * 3))
        self.app.processEvents()

        self.assertGreater(group.sizeHint().height(), before)
        self.assertGreaterEqual(group.maximumHeight(), group.sizeHint().height())

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
            self.assertEqual(workspace.layout_choice.width(), 66, key)

    def test_existing_parameter_tables_use_fixed_scrollable_style(self) -> None:
        self.window.project_toolbar.goal.setCurrentText("Crystal Analysis")
        self.window.module_panel.set_module(self.window.route_modules[0], self.window.orientation_setup_page.controls_panel)
        table = self.window.orientation_setup_page.atom_table
        self.assertEqual(table.height(), PARAM_TABLE_HEIGHT)
        self.assertEqual(table.verticalScrollBarPolicy(), Qt.ScrollBarAsNeeded)

    def test_shared_shell_uses_compact_industrial_density(self) -> None:
        self.window._select_route_module("calibration")
        route_margins = self.window.route_bar.layout.contentsMargins()
        self.assertEqual(
            (route_margins.left(), route_margins.top(), route_margins.right(), route_margins.bottom()),
            (PANEL_MARGIN, PANEL_MARGIN_TIGHT, PANEL_MARGIN, PANEL_MARGIN_TIGHT),
        )
        panel_margins = self.window.module_panel.layout().contentsMargins()
        self.assertEqual(
            (panel_margins.left(), panel_margins.top(), panel_margins.right(), panel_margins.bottom()),
            (PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN),
        )
        self.assertTrue(
            all(
                button.minimumHeight() == ACTION_BUTTON_MIN_HEIGHT
                and button.maximumHeight() == ACTION_BUTTON_MIN_HEIGHT
                for button in self.window.calibration_page.buttons
            )
        )
        self.assertTrue(
            all(button.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored for button in self.window.route_bar.buttons.values())
        )

    def test_preprocessing_uses_one_automatic_show_data_action(self) -> None:
        page = self.window.preprocessing_page

        self.assertEqual(page.show_data_button.text(), "Show Data")
        self.assertFalse(hasattr(page, "diagnostics_button"))
        self.assertFalse(hasattr(page, "show_selected_button"))
        self.assertGreaterEqual(page.memory_budget_mb.value(), 8)

    def test_controls_parameters_share_a_framed_content_surface(self) -> None:
        self.window._select_route_module("calibration")
        surface = self.window.module_panel.controls_stack
        scroll = surface.currentWidget()

        self.assertEqual(surface.objectName(), "moduleControlsSurface")
        self.assertEqual(scroll.objectName(), "moduleControlsScroll")
        self.assertIn(scroll.widget().objectName(), {"moduleControlsContent", SCI_CONTROLS_PANEL})
        margins = scroll.widget().layout().contentsMargins()
        self.assertEqual(
            (margins.left(), margins.top(), margins.right(), margins.bottom()),
            (PANEL_MARGIN + 1, PANEL_MARGIN, PANEL_MARGIN + 1, PANEL_MARGIN),
        )

    def test_bf_df_controls_use_shared_parameter_group_frame(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        self.window._select_route_module("bf_df_preview")
        panel = self.window.bf_df_preview_page.controls_panel

        self.assertIsInstance(panel, QGroupBox)
        self.assertEqual(panel.title(), "BF / DF Preview")
        self.assertEqual(panel.objectName(), "paramForm")
        self.assertTrue(panel.findChildren(QFormLayout))

    def test_checkbox_checked_indicator_uses_green_accent(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "app"
        light_qss = (root / "theme_light.qss").read_text(encoding="utf-8")
        dark_qss = (root / "theme.qss").read_text(encoding="utf-8")

        self.assertIn("QCheckBox::indicator:checked { background: #20a848", light_qss)
        self.assertIn("background: #2db84d;", dark_qss)
        self.assertNotIn("QCheckBox::indicator:checked { background: #2d79b7", light_qss)
        self.assertNotIn("background: #4a9eff;\n    border: 1px solid #4a9eff;", dark_qss)

    def test_parameter_forms_with_more_than_four_parameters_use_zebra_rows(self) -> None:
        self.window._select_route_module("calibration")
        dense_labels = [
            label
            for label in self.window.calibration_page.controls_panel.findChildren(QLabel)
            if label.objectName() == SCI_PROPERTY_LABEL
        ]

        self.assertTrue(any(label.property("rowParity") == "even" for label in dense_labels))
        self.assertTrue(any(label.property("rowParity") == "odd" for label in dense_labels))
        self.assertTrue(all(label.autoFillBackground() for label in dense_labels))

        self.window.project_toolbar.structure.setCurrentText("Crystalline / Bragg-based")
        self.window._select_route_module("data_setup")
        compact_labels = [
            label
            for label in self.window.preprocessing_page.controls_panel.findChildren(QLabel)
            if label.objectName() == SCI_PROPERTY_LABEL
        ]
        self.assertTrue(any(label.property("rowParity") == "even" for label in compact_labels))

    def test_property_grid_name_and_value_cells_share_row_backgrounds(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "app"
        light_qss = (root / "theme_light.qss").read_text(encoding="utf-8")
        dark_qss = (root / "theme.qss").read_text(encoding="utf-8")

        for qss, even, odd in ((light_qss, "#ffffff", "#f2f2f2"), (dark_qss, "#303030", "#383838")):
            self.assertIn('QLabel#propertyGridLabel[rowParity="even"]', qss)
            self.assertIn('QComboBox#propertyGridValue[rowParity="even"]', qss)
            self.assertIn(f"background: {even};", qss)
            self.assertIn('QLabel#propertyGridLabel[rowParity="odd"]', qss)
            self.assertIn('QLineEdit#propertyGridValue[rowParity="odd"]', qss)
            self.assertIn(f"background: {odd};", qss)

    def test_crystal_analysis_workspace_exposes_default_grid_controls(self) -> None:
        self.window.project_toolbar.structure.setCurrentText("Crystalline / Bragg-based")

        for key in ("phase_setup", "orientation_matching"):
            with self.subTest(route=key):
                self.window._select_route_module(key)
                workspace = self.window.viewer_stack.currentWidget().workspace
                self.assertEqual(workspace.layout_choice.currentText(), "4")
                self.assertFalse(workspace.layout_choice.isHidden())
                self.assertFalse(workspace.reset_button.isHidden())
                self.assertEqual(
                    [workspace.layout_choice.itemText(index) for index in range(workspace.layout_choice.count())],
                    ["Auto", "1", "2", "4", "6"],
                )

    def test_crystal_analysis_grid_state_restore_overrides_default(self) -> None:
        self.window.project_coordinator.restore_grid_states(
            {"crystal_cif": {"layout": "2", "page": 0}}
        )

        self.assertEqual(self.window.crystal_cif_page.workspace.grid_state()["layout"], "2")
        self.assertEqual(self.window.crystal_orientation_page.workspace.grid_state()["layout"], "2")
        self.window.crystal_cif_page.workspace.apply_layout_preference("4")

    def test_crystal_analysis_controls_show_representative_units(self) -> None:
        page = self.window.crystal_cif_page

        self.assertEqual(page.voltage.unit_label.text(), "V")
        self.assertEqual(page.k_max.unit_label.text(), "A^-1")
        self.assertEqual(page.zone_step.unit_label.text(), "deg")
        self.assertEqual(page.match_matches.unit_label.text(), "matches")
        self.assertEqual(page.match_min_peaks.unit_label.text(), "peaks")
        self.assertEqual(page.low_confidence.unit_label.text(), "ratio")
        self.assertEqual(page.strain_roi_rx_start.unit_label.text(), "px")

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

    def test_group_box_titles_are_styled_via_qss(self) -> None:
        # Bold styling is now applied globally via theme.qss (QGroupBox { font-weight: bold }),
        # not via imperative font manipulation. Verify the QSS file exists and groups render.
        from pathlib import Path
        qss_path = Path(__file__).resolve().parent.parent / "app" / "theme.qss"
        self.assertTrue(qss_path.exists())
        qss = qss_path.read_text(encoding="utf-8")
        self.assertIn("QGroupBox", qss)
        self.assertIn("font-weight: bold", qss)
        self.assertTrue(
            all(not tabs.tabBar().font().bold() for tabs in self.window.findChildren(QTabWidget))
        )


if __name__ == "__main__":
    unittest.main()
