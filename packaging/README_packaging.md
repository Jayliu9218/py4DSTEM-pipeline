# Packaging Guide

This project keeps development and packaging environments separate:

- `4dstem`: development and debugging
- `.conda\py4dstem-pipeline-packaging`: packaging and release builds

The packaging environment is built **from a clean Python base**, not by cloning
the development environment. This keeps heavy dev-only tools (Jupyter, notebook,
IPython, …) out of the distributable.

## 1. Development

```powershell
conda activate 4dstem
python .\main.py
```

Do not install PyInstaller, Nuitka, or release tools into the development
environment. Create or refresh the separate packaging environment with:

```powershell
.\scripts\setup_packaging_env.ps1
```

The setup script creates a minimal `python=3.11` conda environment, then installs
the project runtime dependencies (pinned in `requirements.txt`) and the
packaging-only tools listed in `requirements.packaging.txt`. It guarantees
py4DSTEM is installed from the GitHub `dev` branch and validates the recorded
Git source and commit before reporting success.

`requirements.txt` declares both the application core (PySide6, numpy, h5py,
pyqtgraph, matplotlib, tifffile — pinned) and py4DSTEM's hard transitive
dependencies (scipy, scikit-*, ncempy, hdf5plugin, gdown, pylops,
colorspacious, pymatgen — floors), which `import py4DSTEM` pulls in eagerly.

The PyInstaller build also launches the packaged application briefly and fails
the build if it exits early. Use `-SkipLaunchTest` only when running in an
environment where desktop applications cannot be launched.

The spec explicitly packages `libexpat.dll` from the dedicated packaging
environment. This avoids accidentally collecting an incompatible DLL from the
system Miniconda installation.

The PyInstaller spec and Nuitka command also explicitly include the application's
visual resources:

- `app/theme_light.qss`
- `app/theme.qss`
- `images/py4DSTEM_logo.png`

These files must remain bundled. If they are missing, the frozen application
can still launch but falls back to the unstyled Qt/Fusion interface, making it
look different from `python main.py`. The packaged-application launch test
checks these resources before starting the executable.

The tested Python reduction backend is the stable runtime path. Packaging does
not require compiled reduction extensions.

## 2. MVP Testing: PyInstaller onedir

```powershell
.\packaging\build_pyinstaller.ps1
```

Output:

```text
dist\pyinstaller\py4DSTEM Pipeline\py4DSTEM Pipeline.exe
```

This is the first distributable build style. It keeps files in a folder, which
is easier to debug than a single executable.

## 3. Group Distribution: PyInstaller onedir + Inno Setup

First build the onedir app:

```powershell
.\packaging\build_pyinstaller.ps1
```

Then open `packaging\inno_setup.iss` in Inno Setup Compiler and build the
installer.

Expected installer output:

```text
packaging\installer\py4DSTEM-Pipeline-Setup-0.1.0.exe
```

## 4. Stable Release Options

### Portable zip

A no-install portable build: a zip of the PyInstaller onedir output. Requires
the PyInstaller build to exist first.

```powershell
.\packaging\build_pyinstaller.ps1
.\packaging\build_portable.ps1
```

Output:

```text
dist\py4DSTEM-Pipeline-v0.1.0-portable.zip
```

### Nuitka

Use this when startup time and a compiled distribution become more important.

```powershell
.\packaging\build_nuitka.ps1
```

Nuitka builds can take much longer than PyInstaller builds.

## Excluded packages

The PyInstaller spec (`packaging\py4dstem_pipeline.spec`) and the Nuitka script
both exclude packages the desktop app never reaches, to shrink the distributable:

- **Notebook / interactive**: notebook, IPython, jupyter\*, ipykernel, ipywidgets,
  nbformat, nbconvert, … (dev-only).
- **py4DSTEM lazy imports**: dask, distributed, mpire — only used by py4DSTEM's
  large-scale parallel processing paths, which the single-machine GUI never
  triggers.
- **Unused heavy deps**: plotly, pandas, sympy, bokeh — pulled in transitively
  by scikit-\*/pymatgen but not touched by the app's py4DSTEM code paths.
- **py4DSTEM optional extras**: tensorflow, cupy, ipyparallel (not installed by
  default).

The py4DSTEM transitive deps that *are* eagerly imported (scipy, scikit-image,
scikit-learn, scikit-optimize, ncempy, hdf5plugin, gdown, pylops,
colorspacious, pymatgen + monty/spglib) **must** stay bundled — excluding them
breaks `import py4DSTEM`.

## Notes

- Build from the project root.
- Keep source notebooks out of the packaged application unless they are
  explicitly needed.
- Test with real `.h5` and `.emd` files after every packaging change.
- If PyInstaller misses a dynamic py4DSTEM dependency, add it to `hiddenimports`
  in `py4dstem_pipeline.spec`.
- The scripts call tools through `python -m ...` where possible so they still
  work when user-site script folders are not on `PATH`.
