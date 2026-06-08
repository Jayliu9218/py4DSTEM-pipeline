from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from app.services.project_state_service import ProjectState, ProjectStateService
from app.services.report_service import ReportService
from app.services.result_registry import ResultRegistry


TEST_OUTPUT = Path(__file__).resolve().parents[1] / ".test-output"


class ProjectServicesTests(unittest.TestCase):
    def test_project_state_round_trip_records_roles_settings_and_params(self) -> None:
        service = ProjectStateService()
        state = ProjectState(
            file_path="sample.h5",
            selected_hdf5_path="/data",
            image_scaling="log",
            image_cmap="viridis",
            cuda_enabled=True,
            recent_export_dir="exports",
            dataset_roles={"target_datacube": "/data", "vacuum_probe": "/probe"},
            page_params={"virtual_detector": {"mode": "Bright Field", "center_x": 12.5}},
        )

        TEST_OUTPUT.mkdir(exist_ok=True)
        path = TEST_OUTPUT / "project_service_project.json"
        service.save(path, state)
        loaded = service.load(path)

        self.assertEqual(loaded.file_path, "sample.h5")
        self.assertEqual(loaded.dataset_roles["target_datacube"], "/data")
        self.assertEqual(loaded.image_cmap, "viridis")
        self.assertTrue(loaded.cuda_enabled)
        self.assertEqual(loaded.page_params["virtual_detector"]["center_x"], 12.5)

    def test_result_registry_lists_overwrites_and_exports_numpy(self) -> None:
        registry = ResultRegistry()
        registry.register("map", "Bragg", np.ones((2, 2)), ("npy",))
        registry.register("map", "Bragg", np.full((2, 2), 3), ("npy",))

        TEST_OUTPUT.mkdir(exist_ok=True)
        path = TEST_OUTPUT / "registry_map.npy"
        registry.export("Bragg/map", path)
        loaded = np.load(path)

        self.assertEqual(len(registry.list_entries()), 1)
        np.testing.assert_array_equal(loaded, np.full((2, 2), 3))

    def test_report_service_generates_markdown_without_results(self) -> None:
        state = ProjectState(
            file_path="sample.h5",
            dataset_roles={"target_datacube": "/data"},
            page_params={"calibration": {"analysis_target": "Strain"}},
        )
        report = ReportService().render_markdown(state, ResultRegistry(), "loaded", "done")

        self.assertIn("# py4DSTEM Pipeline Report", report)
        self.assertIn("target_datacube: /data", report)
        self.assertIn("analysis_target: Strain", report)
        self.assertIn("No results have been registered yet", report)


if __name__ == "__main__":
    unittest.main()
