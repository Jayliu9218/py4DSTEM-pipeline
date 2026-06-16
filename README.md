# py4DSTEM Pipeline

A Windows desktop application for browsing py4DSTEM/HDF5 data and running guided 4D-STEM processing workflows with stage-based review gates.

![Ptychography quick reconstruction](docs/ptychography%20quick%20reconstruction.png)

Built with:

- **PySide6** for the desktop UI
- **PyQtGraph** for interactive image display
- **h5py** for raw HDF5/EMD browsing
- **py4DSTEM** for DataCube, Bragg peak, calibration, orientation, strain, and phase-retrieval algorithms

## How to Use

### First-Time Setup On A New Windows Machine

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and
[Git for Windows](https://git-scm.com/download/win), clone this repository, then
run the progressive setup script from PowerShell:

```powershell
.\scripts\setup_dev_env.ps1
```

The default `All` stage checks prerequisites, creates the repository-local
`.conda\py4dstem-pipeline-dev` Python 3.11 environment, installs the pinned
runtime dependencies, confirms the py4DSTEM Git source, and validates the
environment. The script is idempotent and can be run again after pulling
updates.

Useful follow-up commands:

```powershell
# Check whether Conda and Git are available
.\scripts\setup_dev_env.ps1 -Stage Check

# Refresh dependencies without recreating the environment
.\scripts\setup_dev_env.ps1 -Stage Install

# Validate dependencies and the py4DSTEM source
.\scripts\setup_dev_env.ps1 -Stage Validate

# Run the complete unit-test and compile verification
.\scripts\setup_dev_env.ps1 -Stage Test

# Validate the environment and launch the application
.\scripts\setup_dev_env.ps1 -Stage Validate -Launch
```

To run the application directly after setup:

```powershell
& .\.conda\py4dstem-pipeline-dev\python.exe .\main.py
```

### Basic Workflow

1. Open an HDF5/EMD file and select a DataCube in the Data Browser.
2. Click **Show Data** to activate and assign it as the Target DataCube.
3. Select the Analysis Route and Target from the top toolbar.
4. Complete stages from left to right, reviewing warnings and accepting required gates.
5. Export registered results, CSV data, the project state, or a generated report.

## Current Features

### Data & Preprocessing
- Open `.h5`, `.hdf5`, and `.emd` files
- Browse HDF5 groups and datasets lazily in a tree view
- Select a DataCube and click **Show Data** to activate it as the Target DataCube
- Preview selected 2D diffraction slices and individual 4D scan positions without calculating a full scan overview
- Mean / max diffraction pattern diagnostics
- Hot-pixel detection and correction

### Virtual Imaging
- Bright Field (BF), Annular Dark Field (ADF), and custom annular detectors
- ROI virtual diffraction
- Off-axis dark-field detector

### Bragg & Calibration
- Sigmoid probe-kernel preparation from a vacuum scan ROI
- Bragg peak detection (single pattern, selected positions, or full map)
- Subpixel refinement (poly, multicorr, pixel)
- Origin fitting (max/CoM, plane/parabola/bezier_two); ellipse fitting; pixel-size and rotation calibration
- CBS (convergent-beam) presets for common materials (e.g. Au)

### Crystalline Workflows
- **Orientation Mapping**: CIF or manual crystal setup, orientation plan, single-pattern acceptance gate, full orientation map
- **Strain Mapping**: Bragg vector map → basis selection → reference selection → strain components and quality maps
- **Crystalline Results**: unified results review, quality diagnostics, and export

### Phase Retrieval Workflows
- **DPC / CoM**: segmented virtual detector → CoM preprocessing with rotation/transpose correction → review/accept gate → integrated reconstruction
- **Parallax**: BF disk definition → alignment (Fast/Notebook Quality/Custom presets) → review gate → subpixel refinement, aberration fitting/CTF diagnostics
- **Ptychography**: data & probe setup → geometry calibration → preprocessing acceptance → Quick Reconstruction → QC review/accept → optional parameter optimization → Advanced Reconstruction → export
- **Method Comparison**: side-by-side DPC and Ptychography result viewer

After QC is explicitly accepted, **Apply Best Self-Consistency Value** keeps
Advanced Reconstruction ready. Upstream optimized values are applied by
rebuilding the preprocessing instance inside the background Advanced task;
advanced-only values such as batch size, fix probe, and probe modes are passed
directly to Advanced Reconstruction.

## Guided Workflow

1. Open an HDF5/EMD file and select a DataCube in the Data Browser.
2. Click **Show Data** to activate and assign it as the Target DataCube.
3. Choose an Analysis Route and Target from the top toolbar.
4. Move through the route from left to right, reviewing and accepting scientific gates.
5. Use the Output panel for progress, activity, and warnings; export retained results at the final stage.

Detailed route notes and screenshots are available in
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Interface Gallery

### Data Setup

The Data Browser keeps HDF5 navigation lazy, while **Show Data** activates the
selected DataCube and prepares it for downstream workflows.

![Data setup and lazy DataCube activation](docs/ptychography%20data%20setup.png)

### Ptychography

Quick Reconstruction provides an inexpensive diagnostic pass before QC.
Advanced Reconstruction retains the formal result after QC acceptance and
optional parameter optimization.

| Quick Reconstruction | Advanced Reconstruction |
| --- | --- |
| ![Ptychography quick reconstruction](docs/ptychography%20quick%20reconstruction.png) | ![Ptychography advanced reconstruction](docs/ptychography%20advanced%20reconstruction.png) |

### DPC / CoM

| CoM Preprocessing And Review | Integrated Reconstruction |
| --- | --- |
| ![CoM preprocessing and review](docs/Com%20Prerocessing%20review.png) | ![DPC integrated reconstruction](docs/integrated%20reconstruction.png) |

### Strain Mapping

| Calibration | Strain Analysis |
| --- | --- |
| ![Strain calibration](docs/strain%20calibration.png) | ![Strain analysis](docs/strain%20analysis.png) |

### Export

Registered results can be exported through the compact Export controls,
including batch CSV output for numerical inspection.

| Export Controls | Exported CSV Results |
| --- | --- |
| ![Export controls](docs/data%20export.png) | ![Exported CSV results](docs/data%20csv.png) |

### Shared Infrastructure
- **Industrial light theme** by default, with a dark instrument theme available from the View menu
- Centralized `app/theme.py` color constants and per-widget QSS (`theme.qss` / `theme_light.qss`) applied globally via `QApplication.setStyleSheet()`
- Stage-based workflow with explicit review/accept gates and downstream staleness tracking
- CPU/GPU execution choice with CUDA guidance
- Thread-safe background calculations with live progress reporting
- Dockable UI panels (Data Browser, Controls, Output) with Layout menu toggles and persistent dock state in project files
- Project-state persistence (save/load `.pipeline` files, version 4 with `window_state` for dock layout)
- Scientific report generation
- Result registry for export (NPZ, JSON, PNG, TIFF)

## Environments

Use the progressive setup script to create or refresh the repository-local
development environment:

```powershell
.\scripts\setup_dev_env.ps1
& .\.conda\py4dstem-pipeline-dev\python.exe .\main.py
```

Use the separate repository-local `.conda\py4dstem-pipeline-packaging`
environment for release builds:

```powershell
.\scripts\setup_packaging_env.ps1
.\packaging\build_pyinstaller.ps1
```

The packaging environment is created **from a clean Python 3.11 base** (not by
cloning the development environment), then installs the pinned runtime
dependencies in `requirements.txt` plus the packaging tools in
`requirements.packaging.txt`, and finally ensures py4DSTEM is installed from the
[`dev` branch](https://github.com/py4dstem/py4DSTEM/tree/dev). It disables
Python user-site packages and validates the recorded Git source and commit.
Building from a clean base keeps heavy dev-only tools (Jupyter, notebook,
IPython, …) out of the distributable. PyInstaller builds also run a short
packaged-application launch check by default.

## Project Layout

```text
main.py
app/
  main_window.py                          # Application shell, menu, dialogs, theme switching
  theme.py                                # Centralized color/style constants
  theme.qss                               # Dark theme stylesheet
  theme_light.qss                         # Light theme stylesheet
  controllers/
    application_pages.py                  # Page factory
    route_coordinator.py                  # Route module builder
    project_coordinator.py                # Project save/load
    data_session_controller.py            # HDF5 session state
  pages/
    preprocessing_page.py                 # Hot-pixel detection/correction
    virtual_detector_page.py              # BF/ADF/annular virtual imaging
    bragg_peaks_page.py                   # Bragg peak detection
    calibration_page.py                   # Origin, ellipse, pixel, rotation
    orientation_page.py                   # Orientation mapping (setup + map stages)
    strain_map_page.py                    # Strain map calculation
    crystalline_results_page.py           # Unified results & quality review
    dpc_page.py                           # DPC (segmented/preprocess/reconstruct)
    parallax_page.py                      # Parallax (BF/alignment/advanced)
    ptychography_page.py                  # Ptychography (all stages)
    phase_contrast_page.py                # Legacy phase-contrast overview
    method_comparison_page.py             # DPC vs Ptychography comparison
    bf_df_preview_page.py                 # BF/DF quick preview
    radial_profile_page.py, rdf_page.py,
    fem_page.py, amorphous_strain_page.py # Amorphous routes (in development)
    structural_phase_page.py              # Structural phase (in development)
  services/
    array_reduction.py                    # Streaming HDF5 reductions
    native_array_reduction.py             # Optional native-backend hook
    bragg_strain_service.py               # Bragg detection, strain, calibration params
    hdf5_service.py                       # HDF5 file I/O
    orientation_service.py                # Orientation plan & match
    parallax_service.py                   # Parallax workflow stages
    phase_contrast_service.py             # DPC/legacy phase retrieval
    preprocessing_service.py              # Hot-pixel correction
    project_state_service.py              # .pipeline file persistence
    ptychography_service.py               # Ptychography workflow stages
    py4dstem_service.py                   # py4DSTEM version adapter
    report_service.py                     # Scientific report generation
    result_registry.py                    # Result registration & export
    virtual_detector_service.py           # Virtual detector calculations
    workflow_state.py                     # Step tracking & staleness
  widgets/
    adaptive_image_workspace.py           # Multi-result figure workspace
    hdf5_tree_widget.py                   # HDF5 tree browser
    image_grid_viewer.py                  # Grid-based image display
    image_viewer.py                       # Single-image QtGraph viewer
    log_panel.py                          # Calculation process log
    numeric_line_edit.py                  # Scientific-notation spin box
    pipeline_shell.py                     # Navigation sidebar & dock widgets
    progress_stream.py                    # Worker progress capture
    rgb_image_viewer.py                   # RGB composite viewer
    worker_runner.py                      # Shared background-worker mixin
packaging/
  py4dstem_pipeline.spec
  build_pyinstaller.ps1
  inno_setup.iss
  build_conda_pack.ps1
  build_nuitka.ps1
  README_packaging.md
```

The reduction layer is backend-oriented. The default path is the tested
Python/NumPy implementation; `native_array_reduction.py` is currently a safe
importable stub so packaging and runtime fallback paths can stabilize before any
compiled C++/pybind11 backend is introduced.

## py4DSTEM Resources

- [py4DSTEM source repository](https://github.com/py4dstem/py4DSTEM)
- [py4DSTEM tutorials](https://github.com/py4dstem/py4DSTEM_tutorials)

## Release Roadmap

### 1. Development

Create, validate, and run the independent development environment:

```powershell
.\scripts\setup_dev_env.ps1
& .\.conda\py4dstem-pipeline-dev\python.exe .\main.py
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

Use the portable zip for a no-install distribution (requires the PyInstaller
build from step 2):

```powershell
.\packaging\build_pyinstaller.ps1
.\packaging\build_portable.ps1
```

Output:

```text
dist\py4DSTEM-Pipeline-portable.zip
```

Use Nuitka when a compiled distribution is preferred:

```powershell
.\packaging\build_nuitka.ps1
```

## Packaging Details

See [packaging/README_packaging.md](packaging/README_packaging.md).

## License

This project is distributed under the **GNU General Public License version 3 (GPLv3)**.
See [LICENSE](LICENSE) for the full text.

py4DSTEM is open-source software distributed under a GPLv3 license. It is free
to use, alter, or build on, provided that any work derived from py4DSTEM is also
kept free and open under a GPLv3 license.

## Notes for Development

- Keep py4DSTEM algorithms inside `app/services/`.
- UI pages should be thin: collect parameters, start workers, display results, and log status.
- Use worker threads for expensive calculations so the UI remains responsive.
- **Theme colors**: reference constants from `app/theme.py` instead of hard-coding hex strings. Dark/light QSS files live in `app/theme.qss` and `app/theme_light.qss`.
- **Status colors**: use `Theme.READY`, `Theme.RUNNING`, `Theme.STALE`, `Theme.FAILED`, `Theme.NEUTRAL`, `Theme.TEXT_DISABLED` for semantic consistency across themes.
- Add new pages to `app/controllers/application_pages.py` and add route modules in
  `app/controllers/route_coordinator.py`.
- Add new workflow steps to `app/services/workflow_state.py`.
- Test and debug inside `.conda\py4dstem-pipeline-dev`.
- Keep packaging-only tools and release builds inside
  `.conda\py4dstem-pipeline-packaging`.
