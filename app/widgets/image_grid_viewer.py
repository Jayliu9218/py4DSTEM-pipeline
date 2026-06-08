from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from app.widgets.image_viewer import ImageViewer


class ImageGridViewer(QWidget):
    def __init__(self, rows: int = 2, columns: int = 3) -> None:
        super().__init__()
        self.viewers: list[ImageViewer] = []
        self.labels: list[QLabel] = []
        outer_layout = QVBoxLayout(self)
        self.legend_label = QLabel("Red circles: detected Bragg disks")
        outer_layout.addWidget(self.legend_label)
        layout = QGridLayout()
        outer_layout.addLayout(layout)
        for index in range(rows * columns):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 2, 2, 2)
            label = QLabel(f"Position {index + 1}")
            viewer = ImageViewer()
            panel_layout.addWidget(label)
            panel_layout.addWidget(viewer, 1)
            layout.addWidget(panel, index // columns, index % columns)
            self.labels.append(label)
            self.viewers.append(viewer)

    def set_result(self, index: int, title: str, image, peaks) -> None:
        viewer = self.viewers[index]
        self.labels[index].setText(title)
        viewer.set_image(image)
        viewer.clear_points()
        if len(peaks):
            viewer.set_points(peaks[:, 0], peaks[:, 1], size=7)

    def clear(self) -> None:
        for index, viewer in enumerate(self.viewers):
            self.labels[index].setText(f"Position {index + 1}")
            viewer.clear()
