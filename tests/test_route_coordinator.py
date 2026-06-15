import unittest
from unittest.mock import Mock

from app.controllers.route_coordinator import GOALS_BY_STRUCTURE, RouteCoordinator, build_route_modules
from app.services.workflow_state import WorkflowState, WorkflowStep


class RouteDefinitionTests(unittest.TestCase):
    def test_every_structure_goal_builds_a_valid_route(self) -> None:
        for structure, goals in GOALS_BY_STRUCTURE.items():
            for goal in goals:
                with self.subTest(structure=structure, goal=goal):
                    modules = build_route_modules(structure, goal)
                    keys = [module.key for module in modules]
                    self.assertEqual(keys[0], "data_setup")
                    self.assertEqual(keys[-1], "export")
                    self.assertEqual(len(keys), len(set(keys)))
                    for module in modules:
                        if module.prerequisite:
                            self.assertIn(module.prerequisite, keys)

    def test_focused_routes_keep_expected_stage_order(self) -> None:
        dpc = build_route_modules("Phase Retrieval / Ptychography", "DPC / CoM")
        parallax = build_route_modules("Phase Retrieval / Ptychography", "Parallax")
        ptychography = build_route_modules("Phase Retrieval / Ptychography", "Ptychography")
        orientation = build_route_modules("Crystalline / Bragg-based", "Orientation Mapping")
        strain = build_route_modules("Crystalline / Bragg-based", "Strain Mapping")

        self.assertEqual(
            [module.key for module in dpc],
            ["data_setup", "bf_df_preview", "dpc_segmented", "dpc_preprocess", "dpc", "export"],
        )
        self.assertEqual(
            [module.key for module in parallax],
            [
                "data_setup",
                "parallax_bf",
                "parallax_alignment",
                "parallax_review",
                "parallax_advanced",
                "export",
            ],
        )
        self.assertEqual(
            [module.key for module in ptychography],
            [
                "data_setup", "ptychography_data", "ptychography_geometry", "ptychography_preprocess",
                "ptychography_quick", "ptychography_review", "ptychography_optimization",
                "ptychography_advanced", "export",
            ],
        )
        self.assertEqual(
            [module.key for module in orientation],
            ["data_setup", "virtual_imaging", "bragg_detection", "calibration",
             "orientation_setup", "crystalline_results", "export"],
        )
        self.assertEqual(
            [module.key for module in strain],
            ["data_setup", "virtual_imaging", "bragg_detection", "calibration",
             "crystal_analysis", "crystalline_results", "export"],
        )

    def test_route_states_enforce_data_and_completed_prerequisites(self) -> None:
        ready = [False]
        workflow_state = WorkflowState()
        coordinator = RouteCoordinator(
            toolbar=Mock(),
            route_bar=Mock(),
            module_panel=Mock(),
            viewer_stack=Mock(),
            viewer_pages={},
            workflow_state=workflow_state,
            controls_provider=Mock(),
            workspace_provider=Mock(),
            style_refresher=Mock(),
            data_ready_provider=lambda: ready[0],
        )
        coordinator.modules = build_route_modules("Phase Retrieval / Ptychography", "DPC / CoM")

        states = coordinator.states()
        self.assertEqual(states["data_setup"], "Ready")
        self.assertEqual(states["bf_df_preview"], "Disabled")
        self.assertEqual(states["dpc"], "Disabled")
        self.assertEqual(states["export"], "Disabled")

        ready[0] = True
        states = coordinator.states()
        self.assertEqual(states["bf_df_preview"], "Ready")
        self.assertEqual(states["dpc_preprocess"], "Ready")
        self.assertEqual(states["dpc"], "Disabled")

        workflow_state.mark_completed(WorkflowStep.DPC_REVIEW)
        self.assertEqual(coordinator.states()["dpc"], "Ready")
        workflow_state.mark_completed(WorkflowStep.DPC)
        self.assertEqual(coordinator.states()["export"], "Ready")

        workflow_state.parameters_updated(WorkflowStep.DPC_REVIEW)
        self.assertEqual(coordinator.states()["dpc"], "Disabled")


if __name__ == "__main__":
    unittest.main()
