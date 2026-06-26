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
        panel.process_started("Bragg")
        panel.process_progress("45%")
        panel.process_failed("Bragg", "failed")
        self.assertEqual(panel.progress.value(), 45)
        panel.process_started("Results")
        self.assertEqual((panel.progress.minimum(), panel.progress.maximum()), (0, 0))

    def test_progress_bar_hides_text_and_status_line_shows_latest(self) -> None:
        panel = LogPanel()
        # Progress bar should not display text overlay.
        self.assertFalse(panel.progress.isTextVisible())
        panel.process_started("Long calculation name")
        self.assertEqual((panel.progress.minimum(), panel.progress.maximum()), (0, 0))
        panel.process_progress("nested operation 57%")
        self.assertEqual((panel.progress.minimum(), panel.progress.maximum()), (0, 100))
        # The single-line status should show the latest message.
        self.assertIn("nested operation 57%", panel.status_line.text())
        panel.process_finished("Long calculation name")
        self.assertIn("DONE", panel.status_line.text())

    def test_text_only_progress_uses_indeterminate_loop(self) -> None:
        panel = LogPanel()
        panel.process_started("MIB import")
        panel.process_progress("Reading MIB headers")

        self.assertEqual((panel.progress.minimum(), panel.progress.maximum()), (0, 0))


if __name__ == "__main__":
    unittest.main()
