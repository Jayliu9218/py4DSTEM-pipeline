from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def _load_stylesheet() -> str:
    qss_path = Path(__file__).parent / "app" / "theme.qss"
    try:
        return qss_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def main() -> int:
    app = QApplication([])
    # Fusion gives a consistent cross-platform base for the custom QSS theme.
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    app.setApplicationName("4D-STEM Analysis Pipeline")
    app.setStyleSheet(_load_stylesheet())

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
