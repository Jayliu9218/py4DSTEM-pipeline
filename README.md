# py4DSTEM Pipeline

A minimal Windows desktop application for browsing py4DSTEM/HDF5 data and building an interactive 4D-STEM processing workflow.

The current app is an MVP built with:

- PySide6 for the desktop UI
- PyQtGraph for image display
- h5py for raw HDF5/EMD browsing
- py4DSTEM for DataCube, Bragg peak, calibration, virtual detector, and strain workflows

## Current Features

- Open `.h5`, `.hdf5`, and `.emd` files
- Browse HDF5 groups and datasets in a tree
- Display 2D datasets
- Browse 4D datasets and py4DSTEM DataCube objects
- Select `rx / ry` scan coordinates and display diffraction patterns
- Generate virtual detector images:
  - Bright Field
  - Annular Dark Field
  - Custom Annular Detector
- Run Bragg peak detection on the current diffraction pattern
- Run full BraggVectors calculation
- Check calibration status for origin, ellipse, pixel, and rotation
- Run StrainMap when BraggVectors and calibration are available
- Export virtual detector and strain map results

## Environments

Use the existing `4dstem` Conda environment for development:

```powershell
conda activate 4dstem
python .\main.py
```

Use the separate repository-local `.conda\py4dstem-pipeline-packaging`
environment for release builds:

```powershell
.\scripts\setup_packaging_env.ps1
.\packaging\build_pyinstaller.ps1
```

The packaging environment is cloned from `4dstem`, augmented with the tools in
`requirements.packaging.txt`, then installs py4DSTEM directly from the
[`dev` branch](https://github.com/py4dstem/py4DSTEM/tree/dev). It disables
Python user-site packages and validates the recorded Git source and commit.
PyInstaller builds also run a short packaged-application launch check by
default.

## Project Layout

```text
main.py
app/
  main_window.py
  pages/
    bragg_peaks_page.py
    calibration_page.py
    strain_map_page.py
    virtual_detector_page.py
  services/
    bragg_strain_service.py
    hdf5_service.py
    py4dstem_service.py
    virtual_detector_service.py
  widgets/
    hdf5_tree_widget.py
    image_viewer.py
    log_panel.py
packaging/
  py4dstem_pipeline.spec
  build_pyinstaller.ps1
  inno_setup.iss
  build_conda_pack.ps1
  build_nuitka.ps1
  README_packaging.md
```

## Release Roadmap

### 1. Development

Run directly inside the independent development environment:

```powershell
conda activate 4dstem
python .\main.py
```

### 2. MVP Testing

Build a PyInstaller onedir package:

```powershell
.\packaging\build_pyinstaller.ps1
```

Output:

```text
dist\pyinstaller\py4DSTEM Pipeline\
```

### 3. Group Distribution

Build the PyInstaller onedir package first, then build the installer with Inno Setup:

```powershell
.\packaging\build_pyinstaller.ps1
```

Open this file in Inno Setup Compiler:

```text
packaging\inno_setup.iss
```

### 4. Stable Release

Use conda-pack when environment reproducibility is the priority:

```powershell
.\packaging\build_conda_pack.ps1
```

Use Nuitka when a compiled distribution is preferred:

```powershell
.\packaging\build_nuitka.ps1
```

## Packaging Details

See [packaging/README_packaging.md](packaging/README_packaging.md).

## Notes for Development

- Keep py4DSTEM algorithms inside `app/services/`.
- Keep UI pages thin: they should collect parameters, start workers, display results, and log status.
- Put expensive calculations in workers so the UI remains responsive.
- Test and debug inside `4dstem`.
- Keep packaging-only tools and release builds inside
  `.conda\py4dstem-pipeline-packaging`.
