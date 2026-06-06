from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.services.bragg_strain_service import BraggStrainService
from app.widgets.log_panel import LogPanel


class CalibrationPage(QWidget):
    def __init__(
        self,
        datacube_provider: Callable[[], object | None],
        braggvectors_provider: Callable[[], object | None],
        service: BraggStrainService,
        log_panel: LogPanel,
    ) -> None:
        super().__init__()
        self.datacube_provider = datacube_provider
        self.braggvectors_provider = braggvectors_provider
        self.service = service
        self.log_panel = log_panel

        self.source_label = QLabel("-")
        self.origin_label = QLabel("-")
        self.ellipse_label = QLabel("-")
        self.pixel_label = QLabel("-")
        self.rotate_label = QLabel("-")
        self.complete_label = QLabel("-")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_status)

        form = QFormLayout()
        form.addRow("Source", self.source_label)
        form.addRow("origin", self.origin_label)
        form.addRow("ellipse", self.ellipse_label)
        form.addRow("pixel", self.pixel_label)
        form.addRow("rotate", self.rotate_label)
        form.addRow("complete", self.complete_label)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.refresh_button)
        layout.addStretch(1)

    def refresh_status(self) -> None:
        source = self.braggvectors_provider()
        source_name = "BraggVectors"
        if source is None:
            source = self.datacube_provider()
            source_name = "DataCube"
        if source is None:
            self.source_label.setText("-")
            self.origin_label.setText("missing")
            self.ellipse_label.setText("missing")
            self.pixel_label.setText("missing")
            self.rotate_label.setText("missing")
            self.complete_label.setText("No DataCube or BraggVectors loaded")
            self.log_panel.log("Calibration check: no source loaded.")
            return

        status = self.service.calibration_status(source)
        self.source_label.setText(source_name)
        self.origin_label.setText(status.origin)
        self.ellipse_label.setText(status.ellipse)
        self.pixel_label.setText(status.pixel)
        self.rotate_label.setText(status.rotate)
        self.complete_label.setText("yes" if status.complete else "no")
        self.log_panel.log(f"Calibration check refreshed for {source_name}: complete={status.complete}.")
