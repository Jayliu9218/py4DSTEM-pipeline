from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
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

    def __init__(self) -> None:
        super().__init__()
        self.raw_image: np.ndarray | None = None
        self.flip_x = False
        self._pixmap = QPixmap()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(0, 0)
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
