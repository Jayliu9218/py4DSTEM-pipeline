import unittest

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from app.pages.crystalline_results_page import CrystallineResultsPage
from app.services.result_registry import ResultRegistry


class CrystallineResultsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        registry = ResultRegistry()
        registry.register("Orientation RGB", "orientation", np.ones((2, 2, 3)))
        registry.register("Confidence Gap", "orientation", np.ones((2, 2)))
        registry.register("Candidate overlay", "orientation process", np.ones((2, 2)))
        registry.register("exx", "strain", np.ones((2, 2)))
        registry.register("Residual", "strain process", np.ones((2, 2)))
        self.page = CrystallineResultsPage(registry, QWidget(), lambda: None)

    def test_switches_result_family_and_view_category(self) -> None:
        self.page.refresh_results()
        self.assertEqual([result.title for result in self.page.workspace.results], ["Orientation RGB"])
        self.page.view.setCurrentText("Quality Maps")
        self.assertEqual([result.title for result in self.page.workspace.results], ["Confidence Gap"])
        self.page.family.setCurrentText("Strain")
        self.page.view.setCurrentText("Final Results")
        self.assertEqual([result.title for result in self.page.workspace.results], ["exx"])
        self.assertFalse(self.page.mapping_group.isVisible())

    def test_results_page_does_not_duplicate_export_actions(self) -> None:
        self.assertFalse(hasattr(self.page, "export_button"))
        self.assertFalse(hasattr(self.page, "save_button"))
        self.assertFalse(hasattr(self.page, "report_button"))


if __name__ == "__main__":
    unittest.main()
