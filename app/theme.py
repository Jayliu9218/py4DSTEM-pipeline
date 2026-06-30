"""Centralized theme constants for the default industrial-light interface.

All inline color values across the application should reference these constants
instead of hard-coding hex strings, so the palette stays consistent and tunable
from a single location.
"""

from __future__ import annotations


class Theme:
    # --- Surfaces (industrial light) ---
    PANEL_BG = "#dedede"
    PANEL_BG_ALT = "#c9c9c9"
    PANEL_BORDER = "#8a8a8a"
    CANVAS_BG = "#1a1a1a"       # image viewport (near-black)
    INPUT_BG = "#ffffff"
    GROUP_BG = "#e4e4e4"
    GROUP_TITLE_BG = "#a9a9a9"
    TABLE_HEADER_BG = "#b5b5b5"
    TABLE_GRIDLINES = "#c7c7c7"

    # --- Text ---
    TEXT_PRIMARY = "#111111"
    TEXT_SECONDARY = "#4f4f4f"
    TEXT_DISABLED = "#8a8a8a"

    # --- Accent ---
    ACCENT = "#2d79b7"
    ACCENT_HOVER = "#438bc2"
    ACCENT_PRESSED = "#1f659d"

    # --- Status LEDs ---
    READY = "#4caf50"           # completed / idle-OK (green)
    RUNNING = "#ff9800"         # in-progress (amber)
    STALE = "#ffab40"           # stale / pending re-accept (light amber)
    FAILED = "#f44336"          # error / missing (red)
    NEUTRAL = "#757575"         # not-yet-started (grey)

    # --- Semantic helpers (display labels for LEDs) ---
    LED_CHAR = "●"


PANEL_MARGIN = 4
PANEL_MARGIN_TIGHT = 3
GROUP_SPACING = 4
INPUT_MAX_WIDTH = 280      # compact input control max width
PARAM_GROUP_MAX_HEIGHT = 250
PARAM_ROW_HEIGHT = 18
PARAM_TABLE_HEIGHT = 150
ACTION_BUTTON_MIN_HEIGHT = 24
