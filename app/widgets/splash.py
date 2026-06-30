"""Startup splash screen for the 4D-STEM Analysis pipeline.

Shows the py4DSTEM logo over a flat dark panel with the application title
"4D-STEM Analysis / based on py4DSTEM" while the main window is constructed.
The pixmap is composited at construction time so a real splash pixmap backs
the window (no flicker, works with ``splash.finish(window)``).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from app.theme import Theme


# Splash layout constants (pixels).
_LOGO_WIDTH = 360          # logo is scaled to this width, aspect preserved
_LOGO_TOP_PAD = 28
_TITLE_GAP = 14            # gap between logo and title block
_TITLE_BOTTOM_PAD = 22     # gap between title rule and status message
_SIDE_PAD = 40
_STATUS_HEIGHT = 26


class SplashScreen(QSplashScreen):
    """Branded splash: flat dark panel, py4DSTEM logo, two-line title, status line."""

    def __init__(self, logo_path: Path | None = None) -> None:
        if logo_path is None:
            # app/widgets/splash.py -> project root holds images/
            logo_path = Path(__file__).resolve().parents[2] / "docs" / "py4DSTEM_logo.png"
        self._logo = QPixmap(str(logo_path))
        self._background = QPixmap(self._compose_pixmap())
        super().__init__(self._background)
        # Dark base behind any unpainted area; transparent message default color.
        self.setStyleSheet(f"background: {Theme.PANEL_BG};")

    # -- pixmap composition --------------------------------------------------
    def _compose_pixmap(self) -> QPixmap:
        """Build the static splash pixmap: bg + logo + title + accent rule."""
        logo = self._logo
        logo_w = _LOGO_WIDTH
        logo_h = int(logo.height() * logo_w / max(logo.width(), 1)) if not logo.isNull() else 0

        title_font = QFont("Segoe UI", 19)
        title_font.setBold(True)
        subtitle_font = QFont("Segoe UI", 9)

        # Width is driven by logo + side padding; height by the vertical stack.
        width = logo_w + _SIDE_PAD * 2
        height = (
            _LOGO_TOP_PAD + logo_h + _TITLE_GAP
            + 30            # title line
            + 18            # subtitle line
            + 8             # rule + gap
            + _TITLE_BOTTOM_PAD + _STATUS_HEIGHT
        )

        canvas = QPixmap(width, height)
        canvas.fill(QColor(Theme.PANEL_BG))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Logo, centered horizontally.
        if not logo.isNull():
            scaled = logo.scaled(logo_w, logo_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (width - scaled.width()) // 2
            painter.drawPixmap(x, _LOGO_TOP_PAD, scaled)

        # Title block.
        title_y = _LOGO_TOP_PAD + logo_h + _TITLE_GAP
        painter.setFont(title_font)
        painter.setPen(QColor(Theme.TEXT_PRIMARY))
        painter.drawText(_SIDE_PAD, title_y, width - _SIDE_PAD * 2, 30,
                         Qt.AlignHCenter | Qt.AlignVCenter, "4D-STEM Analysis")

        painter.setFont(subtitle_font)
        painter.setPen(QColor(Theme.TEXT_SECONDARY))
        painter.drawText(_SIDE_PAD, title_y + 30, width - _SIDE_PAD * 2, 18,
                         Qt.AlignHCenter | Qt.AlignVCenter, "based on py4DSTEM")

        # Thin accent rule under the title.
        rule_y = title_y + 30 + 18 + 4
        rule_x = _SIDE_PAD + (width - _SIDE_PAD * 2) // 4
        painter.setPen(QColor(Theme.ACCENT))
        painter.drawLine(rule_x, rule_y, width - rule_x, rule_y)

        painter.end()
        return canvas

    def update_status(self, message: str) -> None:
        """Show a short loading message in the splash footer (bottom center)."""
        self.showMessage(
            message,
            Qt.AlignBottom | Qt.AlignHCenter,
            QColor(Theme.TEXT_SECONDARY),
        )
