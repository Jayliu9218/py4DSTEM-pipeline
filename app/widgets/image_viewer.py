from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ImageViewer(QWidget):
    image_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.scatter_item = pg.ScatterPlotItem()
        self.image_view.getView().addItem(self.scatter_item)
        self.image_view.getView().scene().sigMouseClicked.connect(self._handle_mouse_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)

    def set_image(self, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(f"Image viewer expects a 2D array, got shape {array.shape}.")
        self.image_view.setImage(array, autoLevels=True, autoRange=True)

    def clear(self) -> None:
        self.image_view.clear()
        self.clear_points()

    def set_points(self, x: np.ndarray, y: np.ndarray, size: int = 9) -> None:
        self.scatter_item.setData(
            x=np.asarray(x),
            y=np.asarray(y),
            size=size,
            pen=pg.mkPen("r", width=1.5),
            brush=pg.mkBrush(255, 0, 0, 80),
        )

    def clear_points(self) -> None:
        self.scatter_item.clear()

    def _handle_mouse_clicked(self, event) -> None:
        image_item = self.image_view.getImageItem()
        if image_item is None:
            return

        pos = image_item.mapFromScene(event.scenePos())
        x = int(round(pos.x()))
        y = int(round(pos.y()))
        if x < 0 or y < 0:
            return
        self.image_clicked.emit(x, y)
