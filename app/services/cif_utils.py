from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


def load_py4dstem_crystal_from_cif(py4dstem: Any, cif_path: str | Path) -> Any:
    path = Path(cif_path)
    crystal_cls = py4dstem.process.diffraction.Crystal
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*get_structures is deprecated; use parse_structures.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*fractional coordinates rounded to ideal values.*",
        )
        return crystal_cls.from_CIF(path)
