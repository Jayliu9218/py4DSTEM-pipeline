from app.pages.placeholder_page import _make_placeholder_page

StructuralPhasePage = lambda: _make_placeholder_page(
    "Structural Phase Mapping",
    "This module is reserved for Bragg-vector-based crystalline phase identification.\n\n"
    "It will use calibrated BraggVectors and crystal structure candidates "
    "to produce phase maps, orientation maps per phase, and phase confidence maps.\n\n"
    "Requires calibrated BraggVectors and crystal structure library.\n\n"
    "Status: Coming soon.",
)