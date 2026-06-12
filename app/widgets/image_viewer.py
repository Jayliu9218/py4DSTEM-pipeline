from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QInputDialog, QLabel, QMenu, QMessageBox, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    image_clicked = Signal(int, int)
    roi_changed = Signal(int, int, int, int)
    circle_changed = Signal(float, float, float)
    annulus_changed = Signal(float, float, float, float)
    ellipse_changed = Signal(float, float, float, float, float)
    DEFAULT_SCALING = "log"
    DEFAULT_CMAP = "gray"

    def __init__(self, image_kind: str = "intensity") -> None:
        super().__init__()
        self.scaling = self.DEFAULT_SCALING
        self.colormap = self.DEFAULT_CMAP
        self.image_kind = image_kind
        self.raw_image: np.ndarray | None = None
        self.raw_levels: tuple[float, float] | None = None
        self.rendered_image: np.ndarray | None = None
        self.bragg_sampling = 1
        self.bragg_sampling_provider: Callable[[int], np.ndarray] | None = None
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.getView().setMenuEnabled(False)
        self.coordinate_label = QLabel("x: -, y: -, value: -")
        self.scatter_item = pg.ScatterPlotItem()
        self.message_item = pg.TextItem("", color=(120, 120, 120), anchor=(0.5, 0.5))
        self.image_view.getView().addItem(self.scatter_item)
        self.roi_item = pg.PlotDataItem(pen=pg.mkPen("red", width=2))
        self.image_view.getView().addItem(self.roi_item)
        self.interactive_roi_item: pg.RectROI | None = None
        self.interactive_circle_item: pg.CircleROI | None = None
        self.interactive_annulus_inner_item: pg.CircleROI | None = None
        self.interactive_annulus_outer_item: pg.CircleROI | None = None
        self.interactive_ellipse_item: pg.EllipseROI | None = None
        self._updating_interactive_roi = False
        self._updating_interactive_circle = False
        self._updating_interactive_annulus = False
        self._updating_interactive_ellipse = False
        self.overlay_items: list[pg.GraphicsObject] = []
        self.image_view.getView().addItem(self.message_item)
        self.message_item.setPos(0, 0)
        self.image_view.getView().scene().sigMouseClicked.connect(self._handle_mouse_clicked)
        self.image_view.getView().scene().sigMouseMoved.connect(self._handle_mouse_moved)
        self.image_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_view.customContextMenuRequested.connect(self._show_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)
        layout.addWidget(self.coordinate_label)
        self._apply_colormap()

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

    def set_colormap(self, colormap: str) -> None:
        if colormap not in {
            "gray", "viridis", "magma", "plasma", "inferno", "cividis", "RdBu_r", "PRGn"
        }:
            raise ValueError(f"Unsupported image colormap: {colormap}")
        self.colormap = colormap
        self._apply_colormap()
        if self.raw_image is not None:
            self._render_image(self.raw_image, self.raw_levels)

    def set_bragg_sampling_provider(
        self,
        provider: Callable[[int], np.ndarray] | None,
        sampling: int = 1,
    ) -> None:
        self.bragg_sampling_provider = provider
        self.bragg_sampling = max(int(sampling), 1)

    def _show_context_menu(self, position) -> None:
        self._create_context_menu().exec(self.image_view.mapToGlobal(position))

    def _create_context_menu(self) -> QMenu:
        menu = QMenu(self)
        scaling_menu = menu.addMenu("Scaling")
        scaling_group = QActionGroup(self)
        scaling_group.setExclusive(True)
        for scaling in ("linear", "log"):
            action = scaling_menu.addAction(scaling)
            action.setCheckable(True)
            action.setChecked(self.scaling == scaling)
            action.triggered.connect(
                lambda _checked=False, value=scaling: self.set_scaling(value)
            )
            scaling_group.addAction(action)

        cmap_menu = menu.addMenu("Colormap")
        cmap_group = QActionGroup(self)
        cmap_group.setExclusive(True)
        for cmap in ("gray", "viridis", "magma", "plasma", "inferno", "cividis", "RdBu_r", "PRGn"):
            action = cmap_menu.addAction(cmap)
            action.setCheckable(True)
            action.setChecked(self.colormap == cmap)
            action.triggered.connect(lambda _checked=False, value=cmap: self.set_colormap(value))
            cmap_group.addAction(action)

        if self.bragg_sampling_provider is not None:
            menu.addSeparator()
            sampling_action = menu.addAction(f"BraggVectors Sampling... ({self.bragg_sampling})")
            sampling_action.triggered.connect(self._change_bragg_sampling)
        return menu

    def _change_bragg_sampling(self) -> None:
        sampling, accepted = QInputDialog.getInt(
            self,
            "BraggVectors Sampling",
            "Sampling",
            self.bragg_sampling,
            1,
            64,
            1,
        )
        if not accepted or self.bragg_sampling_provider is None:
            return
        try:
            image = self.bragg_sampling_provider(sampling)
        except Exception as exc:
            QMessageBox.warning(self, "BraggVectors Sampling", str(exc))
            return
        self.bragg_sampling = sampling
        self.set_image(image)

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
            if levels is None:
                low, high = np.nanpercentile(display[np.isfinite(display)], [1, 99])
                safe_levels = self._safe_levels(display, (float(low), float(high)))
            else:
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
        self.clear_interactive_annulus()
        self.message_item.setText(message)
        self.message_item.setPos(0, 0)
        self.raw_image = None
        self.raw_levels = None
        self.rendered_image = None
        self.coordinate_label.setText("x: -, y: -, value: -")

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
        if self.interactive_roi_item is not None:
            self.set_interactive_roi_rect(x0, x1, y0, y1)
            return
        self.roi_item.setData(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
        )

    def clear_roi(self) -> None:
        self.roi_item.clear()
        self.clear_interactive_roi()
        self.clear_interactive_circle()
        self.clear_interactive_ellipse()

    def set_interactive_roi_rect(self, x_start: int, x_end: int, y_start: int, y_end: int) -> None:
        x0, x1 = sorted((int(x_start), int(x_end)))
        y0, y1 = sorted((int(y_start), int(y_end)))
        if x0 == x1 or y0 == y1:
            return
        if self.interactive_roi_item is None:
            self.interactive_roi_item = pg.RectROI(
                [x0, y0],
                [x1 - x0, y1 - y0],
                pen=pg.mkPen("red", width=2),
                movable=True,
                removable=False,
            )
            self.interactive_roi_item.addScaleHandle([1, 1], [0, 0])
            self.interactive_roi_item.addScaleHandle([0, 0], [1, 1])
            self.interactive_roi_item.sigRegionChanged.connect(self._handle_interactive_roi_changed)
            self.image_view.getView().addItem(self.interactive_roi_item)
            self.roi_item.clear()
            return
        self._updating_interactive_roi = True
        self.interactive_roi_item.setPos([x0, y0], update=False)
        self.interactive_roi_item.setSize([x1 - x0, y1 - y0], update=False)
        self.interactive_roi_item.stateChanged(finish=False)
        self._updating_interactive_roi = False

    def clear_interactive_roi(self) -> None:
        if self.interactive_roi_item is None:
            return
        self.image_view.getView().removeItem(self.interactive_roi_item)
        self.interactive_roi_item = None

    def interactive_roi_rect(self) -> tuple[int, int, int, int] | None:
        if self.interactive_roi_item is None:
            return None
        pos = self.interactive_roi_item.pos()
        size = self.interactive_roi_item.size()
        x0 = int(round(float(pos.x())))
        y0 = int(round(float(pos.y())))
        x1 = int(round(float(pos.x() + size.x())))
        y1 = int(round(float(pos.y() + size.y())))
        if self.raw_image is not None:
            x0, x1 = self._clamp_pair(x0, x1, self.raw_image.shape[0])
            y0, y1 = self._clamp_pair(y0, y1, self.raw_image.shape[1])
        else:
            x0, x1 = sorted((max(x0, 0), max(x1, 0)))
            y0, y1 = sorted((max(y0, 0), max(y1, 0)))
        return x0, x1, y0, y1

    def set_interactive_circle(self, x: float, y: float, radius: float) -> None:
        if radius <= 0 or not np.all(np.isfinite([x, y, radius])):
            return
        if self.interactive_circle_item is None:
            self.interactive_circle_item = pg.CircleROI(
                [float(x) - float(radius), float(y) - float(radius)],
                [2 * float(radius), 2 * float(radius)],
                pen=pg.mkPen("c", width=2),
                movable=True,
                removable=False,
                resizable=True,
            )
            self.interactive_circle_item.sigRegionChanged.connect(
                self._handle_interactive_circle_changed
            )
            self.image_view.getView().addItem(self.interactive_circle_item)
            return
        self._updating_interactive_circle = True
        self.interactive_circle_item.setPos(
            [float(x) - float(radius), float(y) - float(radius)],
            update=False,
        )
        self.interactive_circle_item.setSize([2 * float(radius), 2 * float(radius)], update=False)
        self.interactive_circle_item.stateChanged(finish=False)
        self._updating_interactive_circle = False

    def clear_interactive_circle(self) -> None:
        if self.interactive_circle_item is None:
            return
        self.image_view.getView().removeItem(self.interactive_circle_item)
        self.interactive_circle_item = None

    def set_interactive_annulus(
        self,
        x: float,
        y: float,
        inner_radius: float,
        outer_radius: float,
    ) -> None:
        if (
            inner_radius <= 0
            or outer_radius <= inner_radius
            or not np.all(np.isfinite([x, y, inner_radius, outer_radius]))
        ):
            return
        if self.interactive_annulus_outer_item is None:
            self.interactive_annulus_outer_item = pg.CircleROI(
                [x - outer_radius, y - outer_radius],
                [2 * outer_radius, 2 * outer_radius],
                pen=pg.mkPen("c", width=2),
                movable=True,
                removable=False,
                resizable=True,
            )
            self.interactive_annulus_inner_item = pg.CircleROI(
                [x - inner_radius, y - inner_radius],
                [2 * inner_radius, 2 * inner_radius],
                pen=pg.mkPen("c", width=1, style=Qt.DashLine),
                movable=True,
                removable=False,
                resizable=True,
            )
            self.interactive_annulus_outer_item.sigRegionChanged.connect(
                self._handle_interactive_annulus_changed
            )
            self.interactive_annulus_inner_item.sigRegionChanged.connect(
                self._handle_interactive_annulus_changed
            )
            self.image_view.getView().addItem(self.interactive_annulus_outer_item)
            self.image_view.getView().addItem(self.interactive_annulus_inner_item)
        self._set_annulus_geometry(x, y, inner_radius, outer_radius)

    def clear_interactive_annulus(self) -> None:
        view = self.image_view.getView()
        for item in (self.interactive_annulus_inner_item, self.interactive_annulus_outer_item):
            if item is not None:
                view.removeItem(item)
        self.interactive_annulus_inner_item = None
        self.interactive_annulus_outer_item = None

    def interactive_annulus(self) -> tuple[float, float, float, float] | None:
        if self.interactive_annulus_inner_item is None or self.interactive_annulus_outer_item is None:
            return None
        outer_x, outer_y, outer_radius = self._circle_geometry(self.interactive_annulus_outer_item)
        _inner_x, _inner_y, inner_radius = self._circle_geometry(self.interactive_annulus_inner_item)
        return outer_x, outer_y, inner_radius, outer_radius

    def _set_annulus_geometry(
        self, x: float, y: float, inner_radius: float, outer_radius: float
    ) -> None:
        if self.interactive_annulus_inner_item is None or self.interactive_annulus_outer_item is None:
            return
        self._updating_interactive_annulus = True
        for item, radius in (
            (self.interactive_annulus_inner_item, inner_radius),
            (self.interactive_annulus_outer_item, outer_radius),
        ):
            item.setPos([x - radius, y - radius], update=False)
            item.setSize([2 * radius, 2 * radius], update=False)
            item.stateChanged(finish=False)
        self._updating_interactive_annulus = False

    @staticmethod
    def _circle_geometry(item: pg.CircleROI) -> tuple[float, float, float]:
        pos = item.pos()
        size = item.size()
        radius = max(float(size.x()), float(size.y())) * 0.5
        return float(pos.x()) + radius, float(pos.y()) + radius, radius

    def interactive_circle(self) -> tuple[float, float, float] | None:
        if self.interactive_circle_item is None:
            return None
        pos = self.interactive_circle_item.pos()
        size = self.interactive_circle_item.size()
        radius = max(float(size.x()), float(size.y())) * 0.5
        x = float(pos.x()) + radius
        y = float(pos.y()) + radius
        return x, y, radius

    def set_interactive_ellipse(
        self,
        x: float,
        y: float,
        a: float,
        b: float,
        theta: float = 0.0,
    ) -> None:
        if a <= 0 or b <= 0 or not np.all(np.isfinite([x, y, a, b, theta])):
            return
        if self.interactive_ellipse_item is None:
            self.interactive_ellipse_item = pg.EllipseROI(
                [float(x) - float(a), float(y) - float(b)],
                [2 * float(a), 2 * float(b)],
                angle=float(theta),
                pen=pg.mkPen("c", width=2),
                movable=True,
                removable=False,
                resizable=True,
                rotatable=True,
            )
            self.interactive_ellipse_item.sigRegionChanged.connect(
                self._handle_interactive_ellipse_changed
            )
            self.image_view.getView().addItem(self.interactive_ellipse_item)
            return
        self._updating_interactive_ellipse = True
        self.interactive_ellipse_item.setPos(
            [float(x) - float(a), float(y) - float(b)],
            update=False,
        )
        self.interactive_ellipse_item.setSize([2 * float(a), 2 * float(b)], update=False)
        self.interactive_ellipse_item.setAngle(float(theta), update=False)
        self.interactive_ellipse_item.stateChanged(finish=False)
        self._updating_interactive_ellipse = False

    def clear_interactive_ellipse(self) -> None:
        if self.interactive_ellipse_item is None:
            return
        self.image_view.getView().removeItem(self.interactive_ellipse_item)
        self.interactive_ellipse_item = None

    def interactive_ellipse(self) -> tuple[float, float, float, float, float] | None:
        if self.interactive_ellipse_item is None:
            return None
        pos = self.interactive_ellipse_item.pos()
        size = self.interactive_ellipse_item.size()
        a = float(size.x()) * 0.5
        b = float(size.y()) * 0.5
        x = float(pos.x()) + a
        y = float(pos.y()) + b
        theta = float(self.interactive_ellipse_item.angle())
        return x, y, a, b, theta

    def set_circle_overlay(
        self,
        x: float,
        y: float,
        radius: float,
        color: str = "r",
    ) -> None:
        self.clear_overlays()
        self.add_circle_overlay(x, y, radius, color)

    def add_circle_overlay(
        self,
        x: float,
        y: float,
        radius: float,
        color: str = "r",
    ) -> None:
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

    def set_ring_overlay(
        self,
        x: float,
        y: float,
        inner_radius: float,
        outer_radius: float,
        color: str = "r",
    ) -> None:
        self.clear_overlays()
        self.add_ring_overlay(x, y, inner_radius, outer_radius, color)

    def add_ring_overlay(
        self,
        x: float,
        y: float,
        inner_radius: float,
        outer_radius: float,
        color: str = "r",
    ) -> None:
        if inner_radius <= 0 or outer_radius <= inner_radius:
            self.add_circle_overlay(x, y, outer_radius, color)
            return
        if not np.all(np.isfinite([x, y, inner_radius, outer_radius])):
            return
        outer = pg.CircleROI(
            [float(x) - float(outer_radius), float(y) - float(outer_radius)],
            [2 * float(outer_radius), 2 * float(outer_radius)],
            pen=pg.mkPen(color, width=2),
            movable=False,
            removable=False,
            resizable=False,
        )
        inner = pg.CircleROI(
            [float(x) - float(inner_radius), float(y) - float(inner_radius)],
            [2 * float(inner_radius), 2 * float(inner_radius)],
            pen=pg.mkPen(color, width=1, style=Qt.DashLine),
            movable=False,
            removable=False,
            resizable=False,
        )
        outer.setAcceptedMouseButtons(Qt.NoButton)
        inner.setAcceptedMouseButtons(Qt.NoButton)
        self.image_view.getView().addItem(outer)
        self.image_view.getView().addItem(inner)
        self.overlay_items.append(outer)
        self.overlay_items.append(inner)

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
        self.add_ellipse_overlay(x, y, a, b, theta, color)

    def add_ellipse_overlay(
        self,
        x: float,
        y: float,
        a: float,
        b: float,
        theta: float = 0.0,
        color: str = "r",
    ) -> None:
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

    def add_vector_overlays(self, vectors: np.ndarray, color: str = "c") -> None:
        for x, y, dx, dy in np.asarray(vectors, dtype=float):
            if not np.all(np.isfinite([x, y, dx, dy])):
                continue
            line = pg.PlotDataItem(
                [float(x), float(x + dx)],
                [float(y), float(y + dy)],
                pen=pg.mkPen(color, width=2),
            )
            self.image_view.getView().addItem(line)
            self.overlay_items.append(line)

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

    def _apply_colormap(self) -> None:
        if self.image_kind == "color":
            return
        try:
            if self.colormap == "gray":
                cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
            else:
                cmap = pg.colormap.get(self.colormap, source="matplotlib")
            self.image_view.setColorMap(cmap)
        except Exception:
            self.image_view.setColorMap(pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]))

    def _validate_image(self, array: np.ndarray) -> None:
        if self.image_kind == "intensity" and array.ndim != 2:
            raise ValueError(f"Intensity image viewer expects a 2D array, got shape {array.shape}.")
        if self.image_kind == "color":
            if array.ndim != 3 or array.shape[2] not in {3, 4}:
                raise ValueError(
                    "Color image viewer expects an RGB/RGBA array with shape "
                    f"(height, width, 3/4), got shape {array.shape}."
                )

    def _handle_interactive_roi_changed(self, *_args) -> None:
        if self._updating_interactive_roi:
            return
        rect = self.interactive_roi_rect()
        if rect is None:
            return
        x0, x1, y0, y1 = rect
        if x0 == x1 or y0 == y1:
            return
        self.roi_changed.emit(x0, x1, y0, y1)

    def _handle_interactive_circle_changed(self, *_args) -> None:
        if self._updating_interactive_circle:
            return
        circle = self.interactive_circle()
        if circle is None:
            return
        x, y, radius = circle
        if radius <= 0:
            return
        self.circle_changed.emit(x, y, radius)

    def _handle_interactive_annulus_changed(self, changed_item) -> None:
        if self._updating_interactive_annulus:
            return
        if self.interactive_annulus_inner_item is None or self.interactive_annulus_outer_item is None:
            return
        x, y, changed_radius = self._circle_geometry(changed_item)
        inner = self._circle_geometry(self.interactive_annulus_inner_item)[2]
        outer = self._circle_geometry(self.interactive_annulus_outer_item)[2]
        if changed_item is self.interactive_annulus_inner_item:
            inner = min(changed_radius, max(outer - 0.1, 0.1))
        else:
            outer = max(changed_radius, inner + 0.1)
        self._set_annulus_geometry(x, y, inner, outer)
        self.annulus_changed.emit(x, y, inner, outer)

    def _handle_interactive_ellipse_changed(self, *_args) -> None:
        if self._updating_interactive_ellipse:
            return
        ellipse = self.interactive_ellipse()
        if ellipse is None:
            return
        x, y, a, b, theta = ellipse
        if a <= 0 or b <= 0:
            return
        self.ellipse_changed.emit(x, y, a, b, theta)

    def _clamp_pair(self, first: int, second: int, maximum: int) -> tuple[int, int]:
        low, high = sorted((first, second))
        low = min(max(low, 0), max(maximum - 1, 0))
        high = min(max(high, low + 1), max(maximum, 1))
        return low, high

    def _handle_mouse_clicked(self, event) -> None:
        if event.button() == Qt.RightButton:
            return
        image_item = self.image_view.getImageItem()
        if image_item is None:
            return

        pos = image_item.mapFromScene(event.scenePos())
        x = int(round(pos.x()))
        y = int(round(pos.y()))
        if x < 0 or y < 0:
            return
        self.image_clicked.emit(x, y)

    def _handle_mouse_moved(self, scene_pos) -> None:
        image_item = self.image_view.getImageItem()
        if image_item is None or self.raw_image is None:
            self.coordinate_label.setText("x: -, y: -, value: -")
            return
        pos = image_item.mapFromScene(scene_pos)
        x = int(round(pos.x()))
        y = int(round(pos.y()))
        if x < 0 or y < 0 or x >= self.raw_image.shape[0] or y >= self.raw_image.shape[1]:
            self.coordinate_label.setText("x: -, y: -, value: -")
            return
        value = self.raw_image[x, y]
        if np.isscalar(value):
            value_text = f"{float(value):.5g}" if np.isfinite(value) else str(value)
        else:
            value_text = np.array2string(np.asarray(value), precision=4, separator=", ")
        self.coordinate_label.setText(f"x: {x}, y: {y}, value: {value_text}")
