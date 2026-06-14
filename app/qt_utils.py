"""Small Qt utilities shared across the application."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QComboBox


class WheelEventFilter(QObject):
    """Global event filter that suppresses mouse-wheel changes on QComboBox.

    QComboBox scrolls through its items on wheel events by default, which lets
    users accidentally change parameter values while scrolling a dense control
    panel. This filter swallows wheel events on every QComboBox application-wide
    so values can only be changed by an explicit click on the dropdown. The
    combo still opens and selects normally on click; only wheel scrolling is
    blocked. Install once on the QApplication instance (see ``main.py``).
    """

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if event.type() == QEvent.Wheel and isinstance(watched, QComboBox):
            return True
        return super().eventFilter(watched, event)
