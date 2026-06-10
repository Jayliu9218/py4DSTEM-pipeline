from app.pages.placeholder_page import _make_placeholder_page

RadialProfilePage = lambda: _make_placeholder_page(
    "Radial Profile",
    "This module is reserved for radial intensity profiling of diffraction patterns "
    "from amorphous or nanocrystalline materials.\n\n"
    "It will compute azimuthally-averaged radial intensity profiles, "
    "variance maps (FEM), and polar-transformed datacubes.\n\n"
    "Status: Coming soon.",
)