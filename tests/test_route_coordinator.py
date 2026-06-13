import unittest

from app.controllers.route_coordinator import GOALS_BY_STRUCTURE, build_route_modules


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


if __name__ == "__main__":
    unittest.main()
