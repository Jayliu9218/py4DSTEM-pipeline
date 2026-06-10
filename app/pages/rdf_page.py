from app.pages.placeholder_page import _make_placeholder_page

RDFPage = lambda: _make_placeholder_page(
    "RDF",
    "This module is reserved for Radial Distribution Function analysis "
    "of amorphous and disordered materials.\n\n"
    "It will compute structure factors S(k), reduced pair distribution functions g(r), "
    "and full PDFs from radial intensity data.\n\n"
    "Status: Coming soon.",
)