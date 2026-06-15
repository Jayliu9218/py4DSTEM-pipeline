# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


datas = [
    ("..\\app\\theme.qss", "app"),
    ("..\\app\\theme_light.qss", "app"),
    ("..\\images\\py4DSTEM_logo.png", "images"),
]
binaries = [(str(Path(sys.prefix) / "Library" / "bin" / "libexpat.dll"), ".")]
hiddenimports = []

# Packages collected wholesale. py4DSTEM's __init__.py eagerly imports these
# submodules (io/visualize/process/...), so their hard transitive deps must be
# bundled too: scipy, scikit-image/learn/optimize, ncempy, hdf5plugin, gdown,
# pylops, colorspacious, pymatgen (+ monty/spglib). Excluding any of those would
# break `import py4DSTEM` in the frozen app.
for package_name in [
    "py4DSTEM",
    "emdfile",
    "pyqtgraph",
    "h5py",
    "tifffile",
    "matplotlib",
    "pymatgen",
    "monty",
    "spglib",
]:
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("pymatgen")
datas += collect_data_files("pymatgen")


a = Analysis(
    ["..\\main.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # --- Notebook / interactive (dev-only, never needed at runtime) ---
        "notebook", "IPython", "jupyter", "jupyter_client", "jupyter_core",
        "jupyter_server", "jupyterlab", "jupyterlab_server",
        "ipykernel", "ipywidgets", "ipython_genutils",
        "nbformat", "nbconvert", "nbclient",
        # --- Test frameworks ---
        "pytest", "pytest-qt",
        # --- py4DSTEM lazy imports (only triggered by large-scale parallel /
        #     distributed processing paths the desktop GUI never reaches) ---
        "dask", "distributed", "mpire",
        # --- Heavy packages not declared by py4DSTEM and not used by the app.
        #     They sneak in as transitive deps of scikit-* or pymatgen but the
        #     app's py4DSTEM code paths do not touch them. ---
        "plotly", "pandas", "sympy", "bokeh",
        # --- py4DSTEM optional extras (not installed by default) ---
        "tensorflow", "tensorflow_addons",
        "cupy",
        "ipyparallel",
        # --- Misc dev tooling that may leak into the env ---
        "pip", "setuptools", "wheel", "conda_pack", "nuitka",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="py4DSTEM Pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="py4DSTEM Pipeline",
)
