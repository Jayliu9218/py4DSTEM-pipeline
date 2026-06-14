import unittest

from PySide6.QtWidgets import QApplication

from app.widgets.log_panel import LogPanel


class LogPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_current_calculation_progress_never_moves_backward(self) -> None:
        panel = LogPanel()
        panel.process_started("Orientation map")
        panel.process_progress("stage one 60%")
        panel.process_progress("nested stage 20%")
        self.assertEqual(panel.progress.value(), 60)
        panel.process_progress("stage two 80%")
        self.assertEqual(panel.progress.value(), 80)

    def test_failure_keeps_progress_until_next_calculation(self) -> None:
        panel = LogPanel()
        panel.process_started("Probe & Bragg")
        panel.process_progress("45%")
        panel.process_failed("Probe & Bragg", "failed")
        self.assertEqual(panel.progress.value(), 45)
        panel.process_started("Results")
        self.assertEqual(panel.progress.value(), 0)

    def test_progress_track_width_stays_fixed_across_status_changes(self) -> None:
        panel = LogPanel()
        width = panel.progress.width()

        panel.process_started("A long current calculation name")
        panel.process_progress("nested operation with a long status message 57%")
        panel.process_finished("A long current calculation name")

        self.assertEqual(width, panel.PROGRESS_BAR_WIDTH)
        self.assertEqual(panel.progress.minimumWidth(), panel.PROGRESS_BAR_WIDTH)
        self.assertEqual(panel.progress.maximumWidth(), panel.PROGRESS_BAR_WIDTH)


if __name__ == "__main__":
    unittest.main()
