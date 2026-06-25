import types
import unittest

import numpy as np

from app.controllers.route_coordinator import build_route_modules
from app.services.phase_mapping_service import (
    PhaseMappingService,
    PhaseMappingServiceError,
    PhaseMatchParams,
    PhasePlanParams,
)
from app.services.workflow_state import WorkflowState, WorkflowStep


class _OrientationMap:
    def __init__(self, corr_value):
        self.corr = np.full((2, 2), corr_value)


class _Crystal:
    def __init__(self, name, corr_value=0.9):
        self.name = name
        self._corr_value = corr_value
        self.plan_kwargs = {}
        self.matched = False

    def setup_diffraction(self, **_kwargs):
        pass

    def calculate_structure_factors(self, **_kwargs):
        pass

    def orientation_plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def match_orientations(self, _braggvectors, **_kwargs):
        self.matched = True
        return _OrientationMap(self._corr_value)

    def plot_orientation_maps(self, **_kwargs):
        import matplotlib.pyplot as plt
        return np.ones((2, 2, 3, 1)), plt.figure(), None


class _BraggVectors:
    shape = (2, 2)
    calstate = {"center": True, "ellipse": True, "pixel": True, "rotate": True}

    def __init__(self):
        self.cal = np.empty((2, 2), dtype=object)
        self.raw = np.empty((2, 2), dtype=object)


class PhaseMappingServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = PhaseMappingService()
        self.braggvectors = _BraggVectors()

    def test_load_and_remove_crystal_updates_library(self):
        crystal = _Crystal("fcc_ni")
        self.service.context.crystals.append(
            types.SimpleNamespace(name="fcc_ni", crystal=crystal, source="CIF: ni.cif", enabled=True)
        )
        self.assertEqual(len(self.service.crystals), 1)
        self.service.remove_crystal(0)
        self.assertEqual(len(self.service.crystals), 0)

    def test_plan_requires_at_least_one_crystal(self):
        with self.assertRaisesRegex(PhaseMappingServiceError, "at least one crystal"):
            self.service.create_multi_phase_plan(PhasePlanParams())

    def test_match_requires_plan_first(self):
        crystal = _Crystal("fcc_ni")
        self.service.context.crystals.append(
            types.SimpleNamespace(name="fcc_ni", crystal=crystal, source="CIF: ni.cif", enabled=True)
        )
        with self.assertRaisesRegex(PhaseMappingServiceError, "orientation plan first"):
            self.service.match_phases(self.braggvectors, PhaseMatchParams())

    def test_multi_phase_match_produces_phase_id_and_confidence(self):
        fcc = _Crystal("fcc_ni", corr_value=0.9)
        bcc = _Crystal("bcc_fe", corr_value=0.6)
        for crystal in (fcc, bcc):
            self.service.context.crystals.append(
                types.SimpleNamespace(name=crystal.name, crystal=crystal, source=f"CIF: {crystal.name}.cif", enabled=True)
            )
        self.service.create_multi_phase_plan(PhasePlanParams())
        result = self.service.match_phases(self.braggvectors, PhaseMatchParams())
        self.assertEqual(result.phase_id_map.shape, (2, 2))
        self.assertTrue(np.all(result.phase_id_map == 0))
        self.assertEqual(len(result.correlation_maps), 2)
        self.assertEqual(result.phase_names, ["fcc_ni", "bcc_fe"])
        self.assertAlmostEqual(result.phase_fraction["fcc_ni"], 1.0)
        self.assertIn("Phase Map", result.images)
        self.assertIn("Confidence Gap", result.images)

    def test_disable_crystal_excludes_from_matching(self):
        fcc = _Crystal("fcc_ni", corr_value=0.9)
        bcc = _Crystal("bcc_fe", corr_value=0.6)
        for crystal in (fcc, bcc):
            self.service.context.crystals.append(
                types.SimpleNamespace(name=crystal.name, crystal=crystal, source=f"CIF: {crystal.name}.cif", enabled=True)
            )
        self.service.set_crystal_enabled(1, False)
        self.service.create_multi_phase_plan(PhasePlanParams())
        result = self.service.match_phases(self.braggvectors, PhaseMatchParams())
        self.assertEqual(len(result.correlation_maps), 1)
        self.assertIn("at least two enabled crystals", " ".join(result.warnings))

    def test_route_and_state_wiring_for_structural_phase(self):
        modules = build_route_modules("Crystalline / Bragg-based", "Structural Phase Mapping")
        keys = [module.key for module in modules]
        self.assertIn("phase_library", keys)
        self.assertIn("phase_matching", keys)
        state = WorkflowState()
        state.mark_completed(WorkflowStep.STRUCTURAL_PHASE_PLAN)
        state.parameters_updated(WorkflowStep.CALIBRATION_APPLY)
        self.assertTrue(state.is_stale(WorkflowStep.STRUCTURAL_PHASE_PLAN))


if __name__ == "__main__":
    unittest.main()
