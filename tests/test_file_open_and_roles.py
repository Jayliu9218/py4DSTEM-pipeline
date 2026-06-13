from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.services.py4dstem_service import Py4DSTEMService, Py4DSTEMServiceError
from app.widgets.adaptive_image_workspace import FigureResult


class FileOpenAndRolesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        output_dir = Path(__file__).resolve().parents[1] / ".test-output"
        output_dir.mkdir(exist_ok=True)
        self.path = output_dir / "file_open_roles_sample.h5"
        with h5py.File(self.path, "w") as output:
            output.create_dataset("target", data=np.ones((2, 2, 4, 4)))

    def tearDown(self) -> None:
        self.window.close()
        self.path.unlink(missing_ok=True)

    def test_file_open_clears_virtual_workspace_without_failure(self) -> None:
        self.window._open_file_path(str(self.path))

        self.assertIsNotNone(self.window.current_file)
        self.assertEqual(self.window.virtual_detector_page.workspace.results, [])

    def test_current_tree_item_can_be_assigned_as_role(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.window.tree.setCurrentItem(target_item)

        self.window._assign_current_role("target_datacube")

        self.assertEqual(self.window.workflow_state.dataset_roles.target_datacube, "/target")

    def test_initial_target_assignment_preserves_preprocessing_diagnostics(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.window.tree.setCurrentItem(target_item)
        self.window.preprocessing_page.workspace.set_results(
            [FigureResult("Basic diagnostic", np.ones((2, 2)))]
        )

        self.window._assign_current_role("target_datacube")

        self.assertEqual(
            [result.title for result in self.window.preprocessing_page.workspace.results],
            ["Basic diagnostic"],
        )

    def test_replacing_target_clears_preprocessing_diagnostics(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.window.tree.setCurrentItem(target_item)
        self.window.workflow_state.set_dataset_role("target_datacube", "/old-target")
        self.window.preprocessing_page.workspace.set_results(
            [FigureResult("Basic diagnostic", np.ones((2, 2)))]
        )

        self.window._assign_current_role("target_datacube")

        self.assertEqual(self.window.preprocessing_page.workspace.results, [])

    def test_tree_uses_middle_elision_and_full_text_tooltips(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.assertEqual(self.window.tree.textElideMode(), Qt.ElideMiddle)
        self.assertEqual(target_item.toolTip(0), target_item.text(0))

    def test_internal_datacube_data_dataset_resolves_to_parent_group(self) -> None:
        path = self.path.parent / "canonical_datacube.h5"
        with h5py.File(path, "w") as output:
            group = output.create_group("root/datacube")
            group.attrs["python_class"] = "DataCube"
            group.create_dataset("data", data=np.ones((2, 2, 4, 4)))
        datacube = Mock(shape=(2, 2, 4, 4), name="datacube")
        module = Mock()
        module.read.return_value = datacube
        service = Py4DSTEMService()
        service.defer_open_file(path)
        try:
            with patch.object(service, "_py4dstem", return_value=module):
                info = service.load_datacube("/root/datacube/data")
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(info.datapath, "/root/datacube")
        self.assertEqual(module.read.call_args.kwargs["datapath"], "/root/datacube")

    def test_raw_4d_dataset_skips_py4dstem_read(self) -> None:
        module = Mock()
        service = Py4DSTEMService()
        service.defer_open_file(self.path)

        with patch.object(service, "_py4dstem", return_value=module):
            with self.assertRaisesRegex(Py4DSTEMServiceError, "not a py4DSTEM DataCube"):
                service.load_datacube("/target")

        module.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
