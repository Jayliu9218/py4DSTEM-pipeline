from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from app.controllers.application_pages import ApplicationPages
from app.controllers.data_session_controller import DataSessionController
from app.services.hdf5_service import Hdf5Service
from app.services.py4dstem_service import Py4DSTEMService
from app.services.project_state_service import ProjectState
from app.services.workflow_state import WorkflowState
from app.main_window import MainWindow
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_data_session_opens_caches_and_closes_raw_hdf5(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "controller_session.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_dataset("data", data=np.ones((2, 2, 3, 4)))
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())
        try:
            source = session.open_file(path)
            dataset = source["data"]
            first = session.raw_scan_image("/data", dataset)
            second = session.raw_scan_image("/data", dataset)

            self.assertIs(first, second)
            self.assertEqual(session.current_file_path, path)
            closed_path, error = session.close_file()
            self.assertEqual(closed_path, path)
            self.assertIsNone(error)
            self.assertIsNone(session.current_file)
            self.assertIsNone(session.raw_scan_image_cache)
        finally:
            session.close_file()
            path.unlink(missing_ok=True)

    def test_data_session_reuses_recent_diffraction_slices(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "diffraction_cache.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_dataset("data", data=np.ones((2, 2, 3, 4)))
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())
        try:
            source = session.open_file(path)
            dataset = source["data"]
            session.hdf5_service.read_4d_diffraction_pattern = Mock(
                wraps=session.hdf5_service.read_4d_diffraction_pattern
            )

            first = session.diffraction_pattern("/data", dataset, 1, 1)
            second = session.diffraction_pattern("/data", dataset, 1, 1)

            self.assertIs(first, second)
            self.assertEqual(session.hdf5_service.read_4d_diffraction_pattern.call_count, 1)
        finally:
            session.close_file()
            path.unlink(missing_ok=True)

    def test_hdf5_preview_description_is_metadata_only(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "preview_description.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_dataset("slice", data=np.ones((3, 4)))
            cube = output.create_group("cube")
            cube.create_dataset("data", data=np.ones((2, 3, 4, 5)))
        service = Hdf5Service()
        try:
            with h5py.File(path, "r") as source:
                self.assertEqual(service.describe_preview(source["slice"])["kind"], "Diffraction slice")
                description = service.describe_preview(source["cube"])
                self.assertEqual(description, {"kind": "DataCube", "shape": (2, 3, 4, 5)})
                self.assertEqual(service.resolve_4d_dataset(source["cube"]).name, "/cube/data")
        finally:
            path.unlink(missing_ok=True)

    def test_data_session_role_assignment_preserves_initial_preprocess_workspace(self) -> None:
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())
        state = WorkflowState()
        dpc = Mock()
        parallax = Mock()
        clear = Mock()
        registry = Mock()

        session.assign_role(
            "target_datacube",
            "/target",
            workflow_state=state,
            dpc_service=dpc,
            parallax_service=parallax,
            clear_workspaces=clear,
            result_registry=registry,
        )

        clear.assert_called_once_with(exclude_keys={"preprocess"})
        registry.clear.assert_not_called()
        self.assertEqual(state.dataset_roles.target_datacube, "/target")

    def test_application_pages_routes_controls_and_clears_shared_pages_once(self) -> None:
        page = QWidget()
        page.workspace = AdaptiveImageWorkspace()
        page.workspace.clear_results = Mock(wraps=page.workspace.clear_results)
        default = QWidget()
        parallax = QWidget()
        dpc = QWidget()
        pages = ApplicationPages(
            viewer_pages={"one": page, "alias": page},
            route_controls={"data_setup": default},
            crystal_controls={},
            amorphous_controls={},
            dpc_controls={"default": dpc},
            export_controls={"Parallax": parallax, "default": default},
        )
        self.assertIs(pages.controls_for_route("export", "Parallax"), parallax)
        self.assertIs(pages.controls_for_route("dpc", "Method Comparison"), dpc)
        pages.clear_workspaces()
        page.workspace.clear_results.assert_called_once()

    def test_project_results_restore_after_state_application(self) -> None:
        window = MainWindow()
        try:
            state = ProjectState(
                result_entries=[
                    {
                        "key": "Check/map",
                        "name": "map",
                        "category": "Check",
                        "export_formats": ["npy"],
                        "metadata": {},
                    }
                ]
            )
            window.project_coordinator.loaded_project_path = Path("project.json")
            window.project_coordinator.state_service.load_results = Mock(
                return_value={"Check/map": np.ones((2, 2))}
            )

            window.project_coordinator.restore_loaded_results(state)

            self.assertIsNotNone(window.result_registry.get("Check/map"))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
