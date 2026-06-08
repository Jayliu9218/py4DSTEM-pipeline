from __future__ import annotations

import importlib.util
import sys


REQUIRED_MODULES = [
    "PySide6",
    "pyqtgraph",
    "h5py",
    "numpy",
    "py4DSTEM",
    "matplotlib",
    "tifffile",
]


def main() -> int:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        print("All runtime dependencies are available.")
        return 0
    print("Missing runtime dependencies:")
    for name in missing:
        print(f"- {name}")
    print("Install them with: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
