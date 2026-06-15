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

    def test_tree_selection_is_metadata_only_until_preview_is_requested(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)

        with patch.object(self.window.hdf5_service, "read_4d_scan_image") as read_scan:
            with patch.object(
                self.window.hdf5_service,
                "read_4d_diffraction_pattern",
                wraps=self.window.hdf5_service.read_4d_diffraction_pattern,
            ) as read_slice:
                self.window.tree.setCurrentItem(target_item)
                self.assertEqual(self.window.selected_preview_kind, "DataCube")
                self.assertEqual(self.window.preview_status, "Not rendered")
                read_scan.assert_not_called()
                read_slice.assert_not_called()

                self.window.rx_spin.setValue(1)
                self.window.ry_spin.setValue(1)
                read_slice.assert_not_called()

                self.window._preview_selected_node()
                read_scan.assert_not_called()
                read_slice.assert_called_once()
                self.assertIn("[1, 1]", self.window.preview_status)

    def test_two_dimensional_selection_renders_only_after_preview(self) -> None:
        path = self.path.parent / "diffraction_slice.h5"
        with h5py.File(path, "w") as output:
            output.create_dataset("slice", data=np.arange(16).reshape(4, 4))
        try:
            self.window._open_file_path(str(path))
            item = self.window.tree.topLevelItem(0).child(0)
            with patch.object(
                self.window.hdf5_service,
                "read_2d_dataset",
                wraps=self.window.hdf5_service.read_2d_dataset,
            ) as read_slice:
                self.window.tree.setCurrentItem(item)
                read_slice.assert_not_called()
                self.assertEqual(self.window.selected_preview_kind, "Diffraction slice")

                self.window._preview_selected_node()
                read_slice.assert_called_once()
                self.assertEqual(self.window.preview_status, "Rendered diffraction slice")
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

    def test_nested_hdf5_groups_load_children_on_expansion(self) -> None:
        path = self.path.parent / "lazy_tree.h5"
        with h5py.File(path, "w") as output:
            nested = output.create_group("outer/inner")
            nested.create_dataset("slice", data=np.ones((3, 3)))
        try:
            self.window._open_file_path(str(path))
            outer = self.window.tree.topLevelItem(0).child(0)
            self.assertEqual(outer.child(0).text(0), "Loading...")

            outer.setExpanded(True)
            self.app.processEvents()
            inner = outer.child(0)
            self.assertTrue(inner.text(0).startswith("inner"))
            self.assertEqual(inner.child(0).text(0), "Loading...")
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

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
                cached = service.load_datacube("/root/datacube")
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(info.datapath, "/root/datacube")
        self.assertIs(cached, info)
        self.assertEqual(module.read.call_args.kwargs["datapath"], "/root/datacube")
        self.assertEqual(module.read.call_count, 1)

    def test_raw_4d_dataset_skips_py4dstem_read(self) -> None:
        module = Mock()
        service = Py4DSTEMService()
        service.defer_open_file(self.path)

        with patch.object(service, "_py4dstem", return_value=module):
            with self.assertRaisesRegex(Py4DSTEMServiceError, "not a py4DSTEM DataCube"):
                service.load_datacube("/target")

        module.read.assert_not_called()

    def test_raw_scan_image_cache_reuses_and_invalidates_explicit_reduction(self) -> None:
        self.window._open_file_path(str(self.path))
        dataset = self.window.current_file["target"]

        with patch.object(
            self.window.hdf5_service,
            "read_4d_scan_image",
            wraps=self.window.hdf5_service.read_4d_scan_image,
        ) as read_scan:
            self.window._load_raw_4d_dataset("/target", tuple(dataset.shape))
            first = self.window._raw_scan_image("/target", dataset)
            second = self.window._raw_scan_image("/target", dataset)
            self.assertIs(first, second)
            self.assertEqual(read_scan.call_count, 1)

            self.window._clear_raw_scan_image_cache()
            self.window._raw_scan_image("/target", dataset)
            self.assertEqual(read_scan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
