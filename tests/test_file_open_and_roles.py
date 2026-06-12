from __future__ import annotations

import unittest
from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


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

    def test_tree_uses_middle_elision_and_full_text_tooltips(self) -> None:
        self.window._open_file_path(str(self.path))
        target_item = self.window.tree.topLevelItem(0).child(0)
        self.assertEqual(self.window.tree.textElideMode(), Qt.ElideMiddle)
        self.assertEqual(target_item.toolTip(0), target_item.text(0))


if __name__ == "__main__":
    unittest.main()
