from pathlib import Path
from time import monotonic

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.qt_utils import WheelEventFilter
from app.widgets.splash import SplashScreen

# Minimum splash visibility (ms) so the window finishes laying out underneath
# before the splash hides, avoiding a flash of an unready UI.
_SPLASH_MIN_MS = 2500


def _load_stylesheet() -> str:
    # Light theme is the application default; users can switch to dark via View menu.
    qss_path = Path(__file__).parent / "app" / "theme_light.qss"
    try:
        return qss_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _application_icon_path() -> Path:
    # main.py lives at the project root; the logo is in images/.
    return Path(__file__).parent / "images" / "py4DSTEM_logo.png"


def main() -> int:
    app = QApplication([])
    # Fusion gives a consistent cross-platform base for the custom QSS theme.
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    app.setApplicationName("4D-STEM Analysis Pipeline")
    app.setStyleSheet(_load_stylesheet())
    # Swallow mouse-wheel events on every QComboBox so scrolling a control panel
    # cannot accidentally change parameter values.
    app.installEventFilter(WheelEventFilter(app))
    icon_path = _application_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Branded splash covers the (synchronous, heavy) MainWindow construction.
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    splash_start = monotonic()

    window = MainWindow(progress_callback=splash.update_status)
    window.showMaximized()

    # Hold the splash for at least _SPLASH_MIN_MS so the freshly-built window
    # finishes its layout underneath before the splash is dismissed.
    elapsed_ms = (monotonic() - splash_start) * 1000
    remaining = _SPLASH_MIN_MS - elapsed_ms
    if remaining > 0:
        loop = QEventLoop()
        QTimer.singleShot(int(remaining), loop.quit)
        loop.exec()

    splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
