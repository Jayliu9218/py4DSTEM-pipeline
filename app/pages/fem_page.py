from app.pages.placeholder_page import _make_placeholder_page

FEMPage = lambda: _make_placeholder_page(
    "FEM",
    "This module is reserved for Fluctuation Electron Microscopy analysis "
    "of medium-range order in amorphous materials.\n\n"
    "It will compute normalized variance maps V(k), local radial variance maps, "
    "and annular symmetry metrics.\n\n"
    "Status: Coming soon.",
)