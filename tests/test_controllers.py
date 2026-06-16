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


class ArrayLike:
    def __init__(self, shape: tuple[int, ...], dtype=np.float32) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)


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

    def test_raw_scan_cache_key_includes_shape_and_dtype(self) -> None:
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())
        session.current_file_path = Path("sample.h5")
        session.hdf5_service.read_4d_scan_image = Mock(
            side_effect=[
                np.ones((2, 2)),
                np.ones((3, 2)),
                np.ones((3, 2), dtype=np.float32),
            ]
        )

        first = session.raw_scan_image("/data", ArrayLike((2, 2, 3, 4), np.float32))
        second = session.raw_scan_image("/data", ArrayLike((2, 2, 3, 4), np.float32))
        third = session.raw_scan_image("/data", ArrayLike((3, 2, 3, 4), np.float32))
        fourth = session.raw_scan_image("/data", ArrayLike((3, 2, 3, 4), np.float64))

        self.assertIs(first, second)
        self.assertIsNot(second, third)
        self.assertIsNot(third, fourth)
        self.assertEqual(session.hdf5_service.read_4d_scan_image.call_count, 3)
        self.assertEqual(session.raw_scan_image_cache_key.hdf5_path, "/data")
        self.assertEqual(session.raw_scan_image_cache_key.shape, (3, 2, 3, 4))
        self.assertEqual(session.raw_scan_image_cache_key.dtype, "float64")

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

    def test_data_session_tracks_selection_preview_and_active_target(self) -> None:
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())

        selection = session.update_selection(
            "/cube",
            "group",
            preview_kind="DataCube",
            preview_shape=(2, 3, 4, 5),
        )
        self.assertEqual(selection.selected_hdf5_path, "/cube")
        self.assertEqual(session.selected_preview_kind, "DataCube")
        self.assertEqual(session.preview_status, "Not rendered")

        session.mark_preview_rendered("Rendered DataCube diffraction slice [0, 0]")
        session.mark_active_target("/cube", (2, 3, 4, 5), "hdf5")

        self.assertTrue(session.selection.displayed)
        self.assertEqual(session.selection.previewed_hdf5_path, "/cube")
        self.assertEqual(session.selection.last_rendered_path, "/cube")
        self.assertEqual(session.selection.active_target_path, "/cube")
        self.assertEqual(session.current_dataset_path, "/cube")
        self.assertEqual(session.current_4d_source, "hdf5")

        info = session.data_browser_selection_info(
            path="/cube",
            node_type="group",
            shape=(2, 3, 4, 5),
            dtype="-",
            rx=0,
            ry=0,
        )
        self.assertEqual(info["Selected node"], "/cube")
        self.assertEqual(info["Previewed node"], "/cube")
        self.assertEqual(info["Active DataCube"], "/cube")
        self.assertEqual(info["Last rendered"], "/cube")

        session.clear_selection()
        self.assertIsNone(session.selected_hdf5_path)
        self.assertEqual(session.selected_preview_kind, "Not displayable")

    def test_target_bright_field_never_triggers_implicit_reduction(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "overview_cache.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_dataset("data", data=np.ones((2, 2, 3, 4)))
        session = DataSessionController(Hdf5Service(), Py4DSTEMService())
        try:
            source = session.open_file(path)
            dataset = source["data"]
            session.current_4d_source = "hdf5"
            session.current_dataset_path = "/data"
            session.current_dataset_shape = tuple(dataset.shape)
            session.hdf5_service.read_4d_scan_image = Mock(
                wraps=session.hdf5_service.read_4d_scan_image
            )

            self.assertIsNone(session.target_bright_field_image())
            session.hdf5_service.read_4d_scan_image.assert_not_called()
            overview = np.ones(dataset.shape[:2])
            self.assertTrue(session.cache_scan_overview(dataset, overview))
            self.assertIs(session.target_bright_field_image(), session.raw_scan_image_cache)
            self.assertEqual(session.raw_scan_image_cache_key.hdf5_path, "/data")
            self.assertEqual(session.raw_scan_image_cache_key.operation, "scan_overview")
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

    def test_plain_hdf5_group_is_not_read_as_py4dstem_reference(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "plain_group.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_group("4DSTEM_simulation/metadatabundle")
        service = Py4DSTEMService()
        session = DataSessionController(Hdf5Service(), service)
        service.read_datapath = Mock()
        try:
            session.open_file(path)

            source = session.selected_display_source("/4DSTEM_simulation/metadatabundle", None)

            self.assertIsNone(source)
            service.read_datapath.assert_not_called()
        finally:
            session.close_file()
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
