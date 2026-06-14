"""Centralized theme constants for the SEM/FIB-style dark-gray interface.

All inline color values across the application should reference these constants
instead of hard-coding hex strings, so the palette stays consistent and tunable
from a single location.
"""

from __future__ import annotations


class Theme:
    # --- Surfaces (dark gray) ---
    PANEL_BG = "#2b2b2b"        # dock / toolbar / control panel background
    PANEL_BG_ALT = "#333333"    # slightly lighter panel (headers, inputs)
    PANEL_BORDER = "#3c3c3c"    # dividers, panel borders
    CANVAS_BG = "#1a1a1a"       # image viewport (near-black)
    INPUT_BG = "#1e1e1e"        # input fields, tables
    GROUP_BG = "#2f2f2f"        # QGroupBox body
    GROUP_TITLE_BG = "#3a3a3a"  # QGroupBox title chip
    TABLE_HEADER_BG = "#383838"
    TABLE_GRIDLINES = "#404040"

    # --- Text ---
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_DISABLED = "#666666"

    # --- Accent ---
    ACCENT = "#4a9eff"          # selected / active / primary action
    ACCENT_HOVER = "#6bb0ff"
    ACCENT_PRESSED = "#3a8eef"

    # --- Status LEDs ---
    READY = "#4caf50"           # completed / idle-OK (green)
    RUNNING = "#ff9800"         # in-progress (amber)
    STALE = "#ffab40"           # stale / pending re-accept (light amber)
    FAILED = "#f44336"          # error / missing (red)
    NEUTRAL = "#757575"         # not-yet-started (grey)

    # --- Semantic helpers (display labels for LEDs) ---
    LED_CHAR = "●"


PANEL_MARGIN = 6           # compact panel content margin
PANEL_MARGIN_TIGHT = 4     # tighter for toolbars / route bars
GROUP_SPACING = 8          # spacing between QGroupBox sections
INPUT_MAX_WIDTH = 280      # compact input control max width
