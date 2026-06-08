from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ImageViewer(QWidget):
    image_clicked = Signal(int, int)
    DEFAULT_SCALING = "log"

    def __init__(self, image_kind: str = "intensity") -> None:
        super().__init__()
        self.scaling = self.DEFAULT_SCALING
        self.image_kind = image_kind
        self.raw_image: np.ndarray | None = None
        self.raw_levels: tuple[float, float] | None = None
        self.rendered_image: np.ndarray | None = None
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.scatter_item = pg.ScatterPlotItem()
        self.message_item = pg.TextItem("", color=(120, 120, 120), anchor=(0.5, 0.5))
        self.image_view.getView().addItem(self.scatter_item)
        self.roi_item = pg.PlotDataItem(pen=pg.mkPen("y", width=2))
        self.image_view.getView().addItem(self.roi_item)
        self.overlay_items: list[pg.GraphicsObject] = []
        self.image_view.getView().addItem(self.message_item)
        self.message_item.setPos(0, 0)
        self.image_view.getView().scene().sigMouseClicked.connect(self._handle_mouse_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)

    def set_image(
        self,
        image: np.ndarray,
        levels: tuple[float, float] | None = None,
        image_kind: str | None = None,
    ) -> None:
        if image_kind is not None:
            self.set_image_kind(image_kind)
        array = np.asarray(image, dtype=float)
        self._validate_image(array)
        self.raw_image = array
        self.raw_levels = levels
        self._render_image(array, levels)

    def set_image_kind(self, image_kind: str) -> None:
        if image_kind not in {"intensity", "color"}:
            raise ValueError(f"Unsupported image kind: {image_kind}")
        self.image_kind = image_kind
        if self.raw_image is not None:
            self._render_image(self.raw_image, self.raw_levels)

    def set_scaling(self, scaling: str) -> None:
        if scaling not in {"linear", "log"}:
            raise ValueError(f"Unsupported image scaling: {scaling}")
        self.scaling = scaling
        if self.raw_image is not None:
            self._render_image(self.raw_image, self.raw_levels)

    def _render_image(self, array: np.ndarray, levels: tuple[float, float] | None = None) -> None:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            self.clear("Image contains no finite values.")
            return
        display = np.nan_to_num(
            array,
            nan=float(np.nanmin(finite)),
            posinf=float(np.nanmax(finite)),
            neginf=float(np.nanmin(finite)),
        )
        if self._uses_intensity_scaling():
            display = self._scale_display(display)
            safe_levels = self._safe_levels(display, self._scale_levels(levels))
        else:
            safe_levels = None
        self.rendered_image = display
        self.message_item.setText("")
        self.image_view.setImage(display, levels=safe_levels, autoRange=False)
        self._finite_view_range(display.shape[:2])

    def clear(self, message: str = "") -> None:
        self.image_view.setImage(np.zeros((1, 1), dtype=float), levels=(0.0, 1.0), autoRange=False)
        self._finite_view_range((1, 1))
        self.clear_points()
        self.clear_roi()
        self.clear_overlays()
        self.message_item.setText(message)
        self.message_item.setPos(0, 0)
        self.raw_image = None
        self.raw_levels = None
        self.rendered_image = None

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

    def set_roi_rect(self, x_start: int, x_end: int, y_start: int, y_end: int) -> None:
        x0, x1 = sorted((int(x_start), int(x_end)))
        y0, y1 = sorted((int(y_start), int(y_end)))
        if x0 == x1 or y0 == y1:
            self.clear_roi()
            return
        self.roi_item.setData(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
        )

    def clear_roi(self) -> None:
        self.roi_item.clear()

    def set_circle_overlay(
        self,
        x: float,
        y: float,
        radius: float,
        color: str = "r",
    ) -> None:
        self.clear_overlays()
        if radius <= 0 or not np.all(np.isfinite([x, y, radius])):
            return
        circle = pg.CircleROI(
            [float(x) - float(radius), float(y) - float(radius)],
            [2 * float(radius), 2 * float(radius)],
            pen=pg.mkPen(color, width=2),
            movable=False,
            removable=False,
            resizable=False,
        )
        circle.setAcceptedMouseButtons(Qt.NoButton)
        self.image_view.getView().addItem(circle)
        self.overlay_items.append(circle)

    def set_ellipse_overlay(
        self,
        x: float,
        y: float,
        a: float,
        b: float,
        theta: float = 0.0,
        color: str = "r",
    ) -> None:
        self.clear_overlays()
        if a <= 0 or b <= 0 or not np.all(np.isfinite([x, y, a, b, theta])):
            return
        ellipse = pg.EllipseROI(
            [float(x) - float(a), float(y) - float(b)],
            [2 * float(a), 2 * float(b)],
            angle=float(np.degrees(theta)),
            pen=pg.mkPen(color, width=2),
            movable=False,
            removable=False,
            resizable=False,
        )
        ellipse.setAcceptedMouseButtons(Qt.NoButton)
        self.image_view.getView().addItem(ellipse)
        self.overlay_items.append(ellipse)

    def clear_overlays(self) -> None:
        view = self.image_view.getView()
        for item in self.overlay_items:
            view.removeItem(item)
        self.overlay_items = []

    def _safe_levels(
        self,
        array: np.ndarray,
        levels: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if levels is not None and all(np.isfinite(level) for level in levels):
            low, high = float(levels[0]), float(levels[1])
        else:
            low = float(np.nanmin(array))
            high = float(np.nanmax(array))
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        if low == high:
            delta = max(abs(low) * 0.01, 1.0)
            low -= delta
            high += delta
        if low > high:
            low, high = high, low
        return low, high

    def _finite_view_range(self, shape: tuple[int, int]) -> None:
        width = max(int(shape[0]), 1)
        height = max(int(shape[1]), 1)
        view = self.image_view.getView()
        view.setRange(xRange=(0, width), yRange=(0, height), padding=0)

    def _scale_display(self, array: np.ndarray) -> np.ndarray:
        if self.scaling == "linear":
            return array
        return np.sign(array) * np.log1p(np.abs(array))

    def _scale_levels(self, levels: tuple[float, float] | None) -> tuple[float, float] | None:
        if levels is None or self.scaling == "linear":
            return levels
        low, high = levels
        scaled = self._scale_display(np.asarray([low, high], dtype=float))
        return float(scaled[0]), float(scaled[1])

    def _uses_intensity_scaling(self) -> bool:
        return self.image_kind == "intensity"

    def _validate_image(self, array: np.ndarray) -> None:
        if self.image_kind == "intensity" and array.ndim != 2:
            raise ValueError(f"Intensity image viewer expects a 2D array, got shape {array.shape}.")
        if self.image_kind == "color":
            if array.ndim != 3 or array.shape[2] not in {3, 4}:
                raise ValueError(
                    "Color image viewer expects an RGB/RGBA array with shape "
                    f"(height, width, 3/4), got shape {array.shape}."
                )

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
