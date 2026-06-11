from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def _make_placeholder_page(title: str, description: str) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 20, 20, 20)
    heading = QLabel(title)
    heading.setObjectName("moduleTitle")
    heading.setAlignment(Qt.AlignCenter)
    body = QLabel(description)
    body.setWordWrap(True)
    body.setAlignment(Qt.AlignCenter)
    body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    layout.addWidget(heading)
    layout.addWidget(body, 1)

    controls = QWidget()
    ctrl_layout = QVBoxLayout(controls)
    ctrl_layout.setContentsMargins(8, 8, 8, 8)
    ctrl_label = QLabel("Coming soon\n\nThis module will be available in a future version.")
    ctrl_label.setWordWrap(True)
    ctrl_layout.addWidget(ctrl_label)
    ctrl_layout.addStretch(1)
    page.controls_panel = controls

    return page