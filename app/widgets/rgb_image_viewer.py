from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def rgb_array_to_qpixmap(image: np.ndarray, flip_x: bool = False) -> QPixmap:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB image expects shape (height, width, 3), got {array.shape}.")
    if flip_x:
        array = np.fliplr(array)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) >= 0 and float(finite.max()) <= 1:
            array = array * 255
        array = np.nan_to_num(array, nan=0, posinf=255, neginf=0)
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width, channels = array.shape
    qimage = QImage(
        array.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


class RgbImageViewer(QWidget):
    """Display rendered RGB figures without scientific-image transforms."""
    image_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.raw_image: np.ndarray | None = None
        self.flip_x = False
        self._pixmap = QPixmap()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(0, 0)
        self.image_label.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label, 1)

    def set_image(self, image: np.ndarray, flip_x: bool = False) -> None:
        self.raw_image = np.asarray(image).copy()
        self.flip_x = bool(flip_x)
        self._pixmap = rgb_array_to_qpixmap(self.raw_image, self.flip_x)
        self._refresh_pixmap()

    def clear(self, _message: str = "") -> None:
        self.raw_image = None
        self._pixmap = QPixmap()
        self.image_label.clear()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        self.image_label.setPixmap(
            self._pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API name
        if watched is self.image_label and event.type() == QEvent.MouseButtonPress:
            self._emit_image_click(event.position().x(), event.position().y())
        return super().eventFilter(watched, event)

    def _emit_image_click(self, label_x: float, label_y: float) -> None:
        if self.raw_image is None:
            return
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        offset_x = (self.image_label.width() - pixmap.width()) / 2
        offset_y = (self.image_label.height() - pixmap.height()) / 2
        local_x = label_x - offset_x
        local_y = label_y - offset_y
        if not (0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height()):
            return
        height, width = self.raw_image.shape[:2]
        x = min(int(local_x * width / pixmap.width()), width - 1)
        y = min(int(local_y * height / pixmap.height()), height - 1)
        if self.flip_x:
            x = width - 1 - x
        self.image_clicked.emit(x, y)
