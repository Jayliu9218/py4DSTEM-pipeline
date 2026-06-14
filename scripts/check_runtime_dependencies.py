from __future__ import annotations

import importlib.util
import sys


# Application core: the 7 packages the app imports directly.
CORE_MODULES = [
    "PySide6",
    "pyqtgraph",
    "h5py",
    "numpy",
    "py4DSTEM",
    "matplotlib",
    "tifffile",
]

# py4DSTEM hard transitive deps: eagerly imported by `import py4DSTEM` (via its
# io/visualize/process submodule chains), so they must be present even though
# the app never imports them by name. Listed here so a missing one surfaces as a
# clear error during development rather than a late ImportError at runtime.
PY4DSTEM_TRANSITIVE_MODULES = [
    "scipy",
    "skimage",          # scikit-image
    "sklearn",          # scikit-learn
    "skopt",            # scikit-optimize
    "ncempy",
    "hdf5plugin",
    "emdfile",
    "pylops",
    "colorspacious",
    "gdown",
    "pymatgen",
]


def _missing(modules: list[str]) -> list[str]:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def main() -> int:
    missing_core = _missing(CORE_MODULES)
    missing_transitive = _missing(PY4DSTEM_TRANSITIVE_MODULES)

    if not missing_core and not missing_transitive:
        print("All runtime dependencies are available.")
        return 0

    print("Missing runtime dependencies:")
    for name in missing_core:
        print(f"  - {name}  [application core]")
    for name in missing_transitive:
        print(f"  - {name}  [py4DSTEM transitive]")
    print("Install them with: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
