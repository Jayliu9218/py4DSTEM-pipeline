import types
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from app.controllers.route_coordinator import build_route_modules
from app.pages.structural_phase_page import StructuralPhasePage
from app.services.crystal_analysis_service import CrystalAnalysisService
from app.services.phase_mapping_service import PhaseMatchParams, PhasePlanParams
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit


class _OrientationMap:
    def __init__(self, corr):
        self.corr = np.asarray(corr, dtype=float)


class _Crystal:
    def __init__(self, name, corr, strain=None):
        self.name = name
        self._corr = corr
        self._strain = strain
        self.structure_factors_called = False
        self.orientation_plan_called = False

    def setup_diffraction(self, **_kwargs):
        pass

    def calculate_structure_factors(self, **_kwargs):
        self.structure_factors_called = True
        return {"ok": True}

    def orientation_plan(self, **_kwargs):
        self.orientation_plan_called = True

    def match_orientations(self, _braggvectors, **_kwargs):
        return _OrientationMap(self._corr)

    def plot_orientation_maps(self, **_kwargs):
        import matplotlib.pyplot as plt

        rgb = np.zeros((2, 2, 3, 1), dtype=float)
        rgb[..., 0, 0] = 0.25 if self.name.endswith("fcc") else 0.8
        return rgb, plt.figure(), None

    def calculate_strain(self, *_args, **_kwargs):
        if self._strain is None:
            raise AttributeError("strain unavailable")
        return self._strain


class _CrystalNoStrain(_Crystal):
    calculate_strain = None


class _BraggVectors:
    shape = (2, 2)
    calstate = {"center": True, "ellipse": True, "pixel": True, "rotate": True}


class CrystalAnalysisServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _add(self, service, crystal):
        service.context.crystals.append(
            types.SimpleNamespace(
                name=crystal.name,
                crystal=crystal,
                source=f"CIF: {crystal.name}.cif",
                enabled=True,
            )
        )

    def test_crystal_analysis_route_uses_full_stage_order(self):
        modules = build_route_modules("Crystalline / Bragg-based", "Crystal Analysis")
        self.assertEqual(
            [module.key for module in modules],
            [
                "data_setup",
                "bragg_detection",
                "calibration",
                "phase_setup",
                "orientation_matching",
                "strain_analysis",
                "export",
            ],
        )

    def test_crystal_analysis_orientation_and_strain_pages_expose_parameters(self):
        workflow_state = WorkflowState()
        log_panel = LogPanel()
        service = CrystalAnalysisService()
        orientation_page = StructuralPhasePage(
            braggvectors_provider=lambda: None,
            log_panel=log_panel,
            workflow_state=workflow_state,
            service=service,
            stage_mode="orientation_grain",
        )
        strain_page = StructuralPhasePage(
            braggvectors_provider=lambda: None,
            log_panel=log_panel,
            workflow_state=workflow_state,
            service=service,
            stage_mode="strain",
        )

        try:
            self.assertFalse(orientation_page.groups["orientation"].isHidden())
            self.assertFalse(orientation_page.groups["grain"].isHidden())
            self.assertIn("orientation_matches", orientation_page.params_snapshot())
            self.assertFalse(strain_page.groups["strain"].isHidden())
            snapshot = strain_page.params_snapshot()
            self.assertIn("strain_coordinate_rotation", snapshot)
            self.assertIn("strain_reference_mode", snapshot)
        finally:
            orientation_page.close()
            strain_page.close()

    def test_crystal_analysis_page_params_round_trip(self):
        workflow_state = WorkflowState()
        log_panel = LogPanel()
        source = StructuralPhasePage(
            braggvectors_provider=lambda: None,
            log_panel=log_panel,
            workflow_state=workflow_state,
            service=CrystalAnalysisService(),
            stage_mode="library_match",
        )
        restored = StructuralPhasePage(
            braggvectors_provider=lambda: None,
            log_panel=log_panel,
            workflow_state=WorkflowState(),
            service=CrystalAnalysisService(),
            stage_mode="strain",
        )

        try:
            source.run_mode.setCurrentText("Full Dataset")
            source.roi_size.setValue(96)
            source.orientation_matches.setValue(5)
            source.orientation_min_angle.setValue(7.5)
            source.orientation_inversion.setChecked(False)
            source.strain_rotation.setValue(12.5)
            source.strain_max_spacing.setValue(4.5)
            source.strain_reference_mode.setCurrentText("ROI-derived g1/g2")
            source.strain_roi_rx_start.setValue(3)
            source.strain_roi_rx_end.setValue(9)

            restored.apply_params_snapshot(source.params_snapshot())

            self.assertEqual(restored.run_mode.currentText(), "Full Dataset")
            self.assertEqual(restored.roi_size.value(), 96)
            self.assertEqual(restored.orientation_matches.value(), 5)
            self.assertEqual(restored.orientation_min_angle.value(), 7.5)
            self.assertFalse(restored.orientation_inversion.isChecked())
            self.assertEqual(restored.strain_rotation.value(), 12.5)
            self.assertEqual(restored.strain_max_spacing.value(), 4.5)
            self.assertEqual(restored.strain_reference_mode.currentText(), "ROI-derived g1/g2")
            self.assertEqual(restored.strain_roi_rx_start.value(), 3)
            self.assertEqual(restored.strain_roi_rx_end.value(), 9)
        finally:
            source.close()
            restored.close()

    def test_crystal_analysis_plan_changes_stale_crystal_downstream(self):
        workflow_state = WorkflowState()
        workflow_state.mark_completed_many({
            WorkflowStep.CRYSTAL_STRUCTURE_FACTORS,
            WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_STRAIN,
        })
        page = StructuralPhasePage(
            braggvectors_provider=lambda: None,
            log_panel=LogPanel(),
            workflow_state=workflow_state,
            service=CrystalAnalysisService(),
            stage_mode="library_match",
        )

        try:
            page.k_max.setValue(2.0)
            page._invalidate_plan()

            self.assertTrue(workflow_state.is_stale(WorkflowStep.CRYSTAL_STRUCTURE_FACTORS))
            self.assertTrue(workflow_state.is_stale(WorkflowStep.CRYSTAL_PHASE))
            self.assertTrue(workflow_state.is_stale(WorkflowStep.CRYSTAL_ORIENTATION))
            self.assertTrue(workflow_state.is_stale(WorkflowStep.CRYSTAL_STRAIN))
        finally:
            page.close()

    def test_roi_mode_is_centered_and_does_not_resize_source(self):
        service = CrystalAnalysisService()
        self.assertEqual(service.analysis_roi((512, 512)), (192, 320, 192, 320))
        service.set_run_config(types.SimpleNamespace(mode="Full Dataset", roi_size=128))
        self.assertIsNone(service.analysis_roi((512, 512)))

    def test_dual_phase_match_selects_best_phase_and_confidence_gap(self):
        service = CrystalAnalysisService()
        self._add(service, _Crystal("Ti-fcc", [[0.9, 0.2], [0.7, 0.1]]))
        self._add(service, _Crystal("Ti-hcp", [[0.4, 0.8], [0.3, 0.6]]))
        service.calculate_structure_factors(PhasePlanParams())
        service.build_orientation_libraries(PhasePlanParams())

        result = service.match_phases(_BraggVectors(), PhaseMatchParams(low_confidence_threshold=0.25))

        np.testing.assert_array_equal(result.phase_id_map, np.array([[0, 1], [0, 1]]))
        np.testing.assert_allclose(result.confidence_map, np.array([[0.5, 0.6], [0.4, 0.5]]))
        self.assertIn("Composite Phase + Orientation", result.images)

    def test_single_phase_orientation_summary_warns_without_discrimination(self):
        service = CrystalAnalysisService()
        self._add(service, _Crystal("Ti-hcp", [[0.55, 0.55], [0.55, 0.55]]))
        service.build_orientation_libraries(PhasePlanParams())

        result = service.match_phases(_BraggVectors(), PhaseMatchParams(low_confidence_threshold=0.75))

        self.assertTrue(np.all(result.phase_id_map == 0))
        self.assertFalse(np.any(result.images["Low Confidence Mask"]))
        self.assertIn("Phase discrimination requires at least two enabled crystals.", result.warnings)
        self.assertIn("Composite Phase + Orientation", service.orientation_summary().images)

    def test_low_confidence_pixels_generate_warning(self):
        service = CrystalAnalysisService()
        self._add(service, _Crystal("Ti-fcc", [[0.55, 0.51], [0.80, 0.20]]))
        self._add(service, _Crystal("Ti-hcp", [[0.50, 0.49], [0.10, 0.18]]))
        service.build_orientation_libraries(PhasePlanParams())

        result = service.match_phases(_BraggVectors(), PhaseMatchParams(low_confidence_threshold=0.05))

        np.testing.assert_array_equal(result.images["Low Confidence Mask"], np.array([[False, True], [False, True]]))
        self.assertIn("2/4 scan positions are below", " ".join(result.warnings))

    def test_optional_strain_warns_when_api_missing_and_masks_when_available(self):
        service = CrystalAnalysisService()
        self._add(service, _Crystal("Ti-fcc", [[0.9, 0.2], [0.9, 0.2]], strain=np.ones((2, 2))))
        self._add(service, _CrystalNoStrain("Ti-hcp", [[0.1, 0.8], [0.1, 0.8]]))
        service.build_orientation_libraries(PhasePlanParams())
        service.match_phases(_BraggVectors(), PhaseMatchParams())

        result = service.run_strain_analysis(_BraggVectors())

        self.assertIn("Ti-fcc Strain", result.images)
        self.assertTrue(np.isnan(result.images["Ti-fcc Strain"][0, 1]))
        self.assertIn("Ti-hcp: strain calculation is unavailable", " ".join(result.warnings))

    def test_crystal_state_stales_after_calibration_update(self):
        state = WorkflowState()
        state.mark_completed(WorkflowStep.CRYSTAL_PHASE)
        state.parameters_updated(WorkflowStep.CALIBRATION_APPLY)
        self.assertTrue(state.is_stale(WorkflowStep.CRYSTAL_PHASE))

    def test_crystal_stage_groups_do_not_reparent_shared_parameter_widgets(self):
        service = CrystalAnalysisService()
        page = StructuralPhasePage(lambda: None, LogPanel(), WorkflowState(), service=service)
        try:
            plan_widgets = set(page.groups["plan"].findChildren(NumericLineEdit))
            structure_widgets = set(page.groups["structure"].findChildren(NumericLineEdit))
            simulated_widgets = set(page.groups["simulated"].findChildren(NumericLineEdit))

            self.assertIn(page.k_max, plan_widgets)
            self.assertIn(page.zone_step, plan_widgets)
            self.assertNotIn(page.k_max, structure_widgets)
            self.assertNotIn(page.zone_step, simulated_widgets)
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
