from app.pages.placeholder_page import _make_placeholder_page

AmorphousStrainPage = lambda: _make_placeholder_page(
    "Amorphous Strain",
    "This module is reserved for strain mapping of amorphous materials "
    "using elliptical ring fitting.\n\n"
    "It will fit amorphous halos in each diffraction pattern to measure "
    "local strain components (exx, eyy, exy) from elliptical distortion.\n\n"
    "Status: Coming soon.",
)