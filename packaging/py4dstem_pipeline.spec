# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


datas = []
binaries = [(str(Path(sys.prefix) / "Library" / "bin" / "libexpat.dll"), ".")]
hiddenimports = []

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
        "notebook",
        "IPython",
        "jupyter",
        "pytest",
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
