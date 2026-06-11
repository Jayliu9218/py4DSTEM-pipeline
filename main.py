from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication([])
    font = app.font()
    font.setPointSize(font.pointSize() + 2)
    app.setFont(font)
    app.setApplicationName("4D-STEM Analysis Pipeline")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
