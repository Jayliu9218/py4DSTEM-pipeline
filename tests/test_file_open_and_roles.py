from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from app.main_window import MainWindow
from app.services.py4dstem_service import (
    DirectDataCubeImportOptions,
    Py4DSTEMService,
    Py4DSTEMServiceError,
    mib_import_options_from_filename,
    parse_mib_filename_metadata,
)
from app.widgets.adaptive_image_workspace import FigureResult


class RecordingDataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.selections: list[object] = []

    def __getitem__(self, selection):
        self.selections.append(selection)
        return self.data[selection]


class FallbackScanDataCube:
    def __init__(self, data: RecordingDataset) -> None:
        self.data = data
        self.shape = data.shape

    def get_virtual_image(self, *_args, **_kwargs):
        raise AttributeError("native virtual image unavailable")


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

    def test_preview_action_is_prominent_below_subset_controls(self) -> None:
        preview_bar = self.window.findChild(QWidget, "dataPreviewBar")

        self.assertIsNotNone(preview_bar)
        self.assertEqual(self.window.preview_button.objectName(), "previewSelectedButton")
        self.assertEqual(self.window.preview_hint_label.objectName(), "dataPreviewHint")
        self.assertIs(self.window.preview_button.parentWidget(), preview_bar)
        self.assertGreaterEqual(self.window.preview_button.minimumHeight(), 24)

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
                info_text = self._data_info_text()
                self.assertIn("Selected node: /target", info_text)
                self.assertIn("Previewed node: /target", info_text)
                self.assertIn("Active DataCube: /target", info_text)
                self.assertIn("Last rendered: /target", info_text)

    def test_show_data_activates_and_assigns_selected_raw_datacube(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.window.tree.setCurrentItem(target_item)

        source = self.window._get_show_data_source()

        self.assertIsInstance(source, h5py.Dataset)
        self.assertEqual(source.name, "/target")
        self.assertEqual(self.window.current_4d_source, "hdf5")
        self.assertEqual(self.window.current_dataset_path, "/target")
        self.assertEqual(self.window.workflow_state.dataset_roles.target_datacube, "/target")

    def _data_info_text(self) -> str:
        root = self.window.tree.info_root_item
        self.assertIsNotNone(root)
        lines: list[str] = []
        for group_index in range(root.childCount()):
            group = root.child(group_index)
            for child_index in range(group.childCount()):
                lines.append(group.child(child_index).text(0))
        return "\n".join(lines)

    def test_show_data_activates_py4dstem_datacube_without_preview(self) -> None:
        path = self.path.parent / "show_data_datacube.h5"
        with h5py.File(path, "w") as output:
            group = output.create_group("cube")
            group.attrs["python_class"] = "DataCube"
            group.create_dataset("data", data=np.ones((2, 2, 4, 4)))
        datacube = Mock(shape=(2, 2, 4, 4), name="cube")
        module = Mock()
        module.read.return_value = datacube
        try:
            self.window._open_file_path(str(path))
            self.window._handle_node_selected("/cube", "group")
            with (
                patch.object(self.window.py4dstem_service, "_py4dstem", return_value=module),
                patch.object(self.window.calibration_page, "refresh_status"),
            ):
                source = self.window._get_show_data_source()

            self.assertIs(source, datacube)
            self.assertIs(self.window._get_py4dstem_datacube(), datacube)
            self.assertEqual(self.window.current_dataset_path, "/cube")
            self.assertEqual(self.window.workflow_state.dataset_roles.target_datacube, "/cube")
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

    def test_mib_file_opens_as_direct_py4dstem_datacube(self) -> None:
        path = self.path.parent / "direct_datacube.mib"
        path.write_bytes(b"mib")
        data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        datacube = Mock(shape=data.shape, name="direct_datacube", data=data)
        module = Mock()
        module.import_file.return_value = datacube
        try:
            with (
                patch.object(self.window.py4dstem_service, "_py4dstem", return_value=module),
                patch.object(self.window.calibration_page, "refresh_status"),
            ):
                self.window._open_file_path(str(path))

            self.assertIsNone(self.window.current_file)
            self.assertIs(self.window._get_py4dstem_datacube(), datacube)
            self.assertEqual(self.window.current_4d_source, "py4dstem")
            self.assertEqual(self.window.current_dataset_path, str(path))
            self.assertEqual(self.window.workflow_state.dataset_roles.target_datacube, str(path))
            self.assertEqual(self.window.tree.headerItem().text(0), "Data source")
            self.assertEqual(self.window.selected_preview_kind, "DataCube")
            self.assertTrue(self.window.preview_button.isEnabled())
            module.import_file.assert_called_once_with(
                str(path),
                mem="MEMMAP",
                scan=(512, 512),
            )

            self.window._preview_selected_node()

            np.testing.assert_array_equal(self.window.diffraction_viewer.raw_image, data[0, 0, :, :])
            self.assertIn("[0, 0]", self.window.preview_status)
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

    def test_mib_filename_metadata_prefills_import_defaults(self) -> None:
        path = self.path.parent / (
            "1_512x512_ss15.63nm_0.55ms_c2 50um_CL91mm_"
            "0.75mrad_spot7_0.022nA_GL3_mag12500k_12b 0913.mib"
        )

        metadata = parse_mib_filename_metadata(path)
        options = mib_import_options_from_filename(path)

        self.assertEqual(metadata["scan_shape"], (512, 512))
        self.assertEqual(metadata["scan_step_nm"], 15.63)
        self.assertEqual(metadata["dwell_ms"], 0.55)
        self.assertEqual(metadata["camera_length_mm"], 91.0)
        self.assertEqual(metadata["convergence_mrad"], 0.75)
        self.assertEqual(metadata["beam_current_nA"], 0.022)
        self.assertEqual(metadata["bit_depth"], 12)
        self.assertEqual(options.scan_shape, (512, 512))
        self.assertEqual(options.preview_scan_stride, 4)

    def test_mib_open_applies_preview_stride_and_logs_metadata(self) -> None:
        path = self.path.parent / (
            "1_512x512_ss15.63nm_0.55ms_c2 50um_CL91mm_"
            "0.75mrad_spot7_0.022nA_GL3_mag12500k_12b 0913.mib"
        )
        path.write_bytes(b"mib")
        datacube = Mock(shape=(2, 3, 4, 5), name="direct_datacube", data=np.ones((2, 3, 4, 5)))
        module = Mock()
        module.import_file.return_value = datacube
        try:
            with (
                patch.object(self.window.py4dstem_service, "_py4dstem", return_value=module),
                patch.object(self.window.calibration_page, "refresh_status"),
            ):
                self.window._open_file_path(
                    str(path),
                    import_options=mib_import_options_from_filename(path),
                )

            self.assertEqual(int(self.window.preprocessing_page.preview_scan_stride.value()), 4)
            self.assertIn("preview_stride=4", self.window.log_panel.status_line.text())
            self.assertIn("scan=(512, 512)", self.window.log_panel.process_log.toPlainText())
            self.assertIn("scan=(512, 512)", self.window.log_panel.event_log.toPlainText())
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

    def test_mib_full_dataset_option_updates_crystal_analysis_controls(self) -> None:
        path = self.path.parent / "full_dataset_direct_datacube.mib"
        path.write_bytes(b"mib")
        datacube = Mock(shape=(2, 3, 4, 5), name="direct_datacube", data=np.ones((2, 3, 4, 5)))
        module = Mock()
        module.import_file.return_value = datacube
        try:
            with (
                patch.object(self.window.py4dstem_service, "_py4dstem", return_value=module),
                patch.object(self.window.calibration_page, "refresh_status"),
            ):
                self.window._open_file_path(
                    str(path),
                    import_options=DirectDataCubeImportOptions(
                        scan_shape=(512, 512),
                        mem_mode="MEMMAP",
                        roi_tuning_mode=False,
                    ),
                )

            self.assertEqual(self.window.crystal_cif_page.run_mode.currentText(), "Full Dataset")
            self.assertEqual(self.window.crystal_phase_page.run_mode.currentText(), "Full Dataset")
            self.assertIsNone(self.window.phase_mapping_service.analysis_roi((512, 512)))
        finally:
            self.window._close_current_file()
            path.unlink(missing_ok=True)

    def test_direct_datacube_scan_image_fallback_uses_bounded_blocks(self) -> None:
        data = np.arange(5 * 4 * 3 * 6, dtype=np.float32).reshape(5, 4, 3, 6)
        source = RecordingDataset(data)
        service = Py4DSTEMService()
        service.datacube = FallbackScanDataCube(source)

        scan_image = service.get_scan_image()

        np.testing.assert_allclose(scan_image, data.sum(axis=(2, 3)))
        self.assertTrue(source.selections)
        for selection in source.selections:
            self.assertIsInstance(selection, tuple)
            first_axis = selection[0]
            self.assertIsInstance(first_axis, slice)
            self.assertLess(first_axis.stop - first_axis.start, data.shape[0])

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

    def test_plain_hdf5_group_skips_py4dstem_reference_read(self) -> None:
        path = self.path.parent / "plain_metadata_group.h5"
        with h5py.File(path, "w") as output:
            output.create_group("4DSTEM_simulation/metadatabundle")
        module = Mock()
        service = Py4DSTEMService()
        service.defer_open_file(path)
        try:
            with patch.object(service, "_py4dstem", return_value=module):
                with self.assertRaisesRegex(Py4DSTEMServiceError, "not a py4DSTEM object"):
                    service.read_datapath("/4DSTEM_simulation/metadatabundle")
        finally:
            path.unlink(missing_ok=True)

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
