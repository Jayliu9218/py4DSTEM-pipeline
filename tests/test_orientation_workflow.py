import types
import unittest

import numpy as np

from app.controllers.route_coordinator import build_route_modules
from app.services.orientation_service import (
    ManualCrystalParams,
    OrientationMapParams,
    OrientationPlanParams,
    OrientationService,
    SinglePatternMatchParams,
)
from app.services.workflow_state import WorkflowState, WorkflowStep


class _Cell:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _Grid:
    shape = (2, 2)

    def __init__(self):
        self.cells = [[_Cell([[1, 0, 2], [0, 1, 1]]) for _ in range(2)] for _ in range(2)]

    def __getitem__(self, index):
        return self.cells[index[0]][index[1]]


class _Orientation:
    corr = np.asarray([0.9, 0.6, 0.2])
    zone_axis = np.asarray([[0, 0, 1]])
    in_plane_rotation = np.asarray([12.0])


class _OrientationMap:
    corr = np.ones((2, 2))
    corr_gap = np.ones((2, 2)) * 0.2


class _Crystal:
    def __init__(self):
        self.plan_kwargs = {}
        self.single_peaks = None
        self.plot_method = None

    def setup_diffraction(self, **_kwargs):
        pass

    def calculate_structure_factors(self, **_kwargs):
        pass

    def orientation_plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def match_single_pattern(self, peaks, **_kwargs):
        self.single_peaks = peaks
        return _Orientation()

    def generate_diffraction_pattern(self, _orientation, ind_orientation=0, **_kwargs):
        return _Cell([[ind_orientation + 1, 0, 1], [0, ind_orientation + 1, 1]])

    def match_orientations(self, _peaks, **_kwargs):
        return _OrientationMap()

    def plot_orientation_maps(self, **_kwargs):
        import matplotlib.pyplot as plt
        self.plot_method = "general"
        return np.ones((2, 2, 3, 1)), plt.figure(), None

    def plot_fiber_orientation_maps(self, *_args, **_kwargs):
        import matplotlib.pyplot as plt
        self.plot_method = "fiber"
        return np.ones((2, 2, 3, 1)), plt.figure(), None


class OrientationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.crystal = _Crystal()
        self.service = OrientationService()
        self.service.crystal = self.crystal
        self.braggvectors = types.SimpleNamespace(
            cal=_Grid(), raw=_Grid(), shape=(2, 2),
            calstate={"center": True, "ellipse": True, "pixel": True, "rotate": True},
        )

    def test_manual_crystal_validates_atom_rows_and_uses_unitcell_factory(self):
        captured = {}

        class CrystalFactory:
            @staticmethod
            def from_unitcell_parameters(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _Crystal()

        self.service._py4dstem = lambda: types.SimpleNamespace(
            process=types.SimpleNamespace(diffraction=types.SimpleNamespace(Crystal=CrystalFactory))
        )
        summary = self.service.create_manual_crystal(
            ManualCrystalParams(elements=("Au",), positions=((0, 0, 0),))
        )
        self.assertIn("Manual cubic", summary)
        self.assertEqual(captured["kwargs"]["lattice_type"], "cubic")
        with self.assertRaisesRegex(Exception, "one element"):
            self.service.create_manual_crystal(
                ManualCrystalParams(elements=("Au", "Ag"), positions=((0, 0, 0),))
            )

    def test_fiber_and_general_plans_forward_expected_ranges(self):
        self.service.create_plan_stage(
            OrientationPlanParams(mode="Fiber", fiber_axis=(0, 0, 1), fiber_angles=(0, 180))
        )
        self.assertEqual(self.crystal.plan_kwargs["zone_axis_range"], "fiber")
        self.assertEqual(self.crystal.plan_kwargs["fiber_axis"], [0, 0, 1])
        self.service.create_plan_stage(OrientationPlanParams(mode="General 3D"))
        self.assertEqual(self.crystal.plan_kwargs["zone_axis_range"], "auto")
        self.assertNotIn("fiber_axis", self.crystal.plan_kwargs)

    def test_single_review_uses_calibrated_peaks_and_requires_acceptance(self):
        self.service.create_plan_stage(OrientationPlanParams())
        result = self.service.review_single_pattern(
            self.braggvectors, SinglePatternMatchParams(scan_x=1, scan_y=0)
        )
        self.assertIs(self.crystal.single_peaks, self.braggvectors.cal[1, 0])
        self.assertIn("Candidate 1 overlay", result.images)
        self.assertAlmostEqual(result.metrics["confidence_gap"], 0.3)
        with self.assertRaisesRegex(Exception, "Accept the single-pattern"):
            self.service.match_map(self.braggvectors, OrientationMapParams())

    def test_full_map_selects_plotter_and_builds_quality_maps(self):
        self.service.create_plan_stage(OrientationPlanParams(mode="Fiber"))
        self.service.review_single_pattern(self.braggvectors, SinglePatternMatchParams())
        self.service.accept_single_review()
        result = self.service.match_map(self.braggvectors, OrientationMapParams())
        self.assertEqual(self.crystal.plot_method, "fiber")
        self.assertIn("Orientation RGB", result.images)
        self.assertIn("Confidence Gap", result.images)
        self.assertIn("Peak Count", result.images)
        self.assertIn("Low Confidence Mask", result.images)

    def test_route_and_state_follow_shared_crystalline_acceptance_gate(self):
        modules = build_route_modules("Crystalline / Bragg-based", "Orientation Mapping")
        self.assertEqual(
            [module.key for module in modules],
            ["data_setup", "virtual_imaging", "bragg_detection", "calibration",
             "orientation_setup", "crystalline_results", "export"],
        )
        state = WorkflowState()
        state.mark_completed_many({
            WorkflowStep.ORIENTATION_PLAN, WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT, WorkflowStep.ORIENTATION_MATCH,
        })
        state.parameters_updated(WorkflowStep.ORIENTATION_REVIEW)
        self.assertTrue(state.is_stale(WorkflowStep.ORIENTATION_REVIEW_ACCEPT))
        self.assertTrue(state.is_stale(WorkflowStep.ORIENTATION_MATCH))


if __name__ == "__main__":
    unittest.main()
