# Packaging Guide

This project keeps development and packaging environments separate:

- `4dstem`: development and debugging
- `.conda\py4dstem-pipeline-packaging`: packaging and release builds

## 1. Development

```powershell
conda activate 4dstem
python .\main.py
```

Do not install PyInstaller, Nuitka, or conda-pack into the development
environment. Create or refresh the separate packaging environment with:

```powershell
.\scripts\setup_packaging_env.ps1
```

The build scripts fail immediately if this environment is missing or a build
command fails.

The setup script clones the `4dstem` runtime locally, installs the project
runtime dependencies and packaging-only tools listed in
`requirements.packaging.txt`, and then replaces py4DSTEM with the latest commit from
`https://github.com/py4dstem/py4DSTEM.git@dev`. It validates the recorded Git
source and commit before reporting success.

The PyInstaller build also launches the packaged application briefly and fails
the build if it exits early. Use `-SkipLaunchTest` only when running in an
environment where desktop applications cannot be launched.

The spec explicitly packages `libexpat.dll` from the dedicated packaging
environment. This avoids accidentally collecting an incompatible DLL from the
system Miniconda installation.

## 2. MVP Testing: PyInstaller onedir

```powershell
.\packaging\build_pyinstaller.ps1
```

Output:

```text
dist\pyinstaller\py4DSTEM Pipeline\py4DSTEM Pipeline.exe
```

This is the first distributable build style. It keeps files in a folder, which is easier to debug than a single executable.

## 3. Group Distribution: PyInstaller onedir + Inno Setup

First build the onedir app:

```powershell
.\packaging\build_pyinstaller.ps1
```

Then open `packaging\inno_setup.iss` in Inno Setup Compiler and build the installer.

Expected installer output:

```text
packaging\installer\py4DSTEM-Pipeline-Setup-0.1.0.exe
```

## 4. Stable Release Options

### Conda-pack

Use this when reproducibility of the scientific environment matters most.

```powershell
.\packaging\build_conda_pack.ps1
```

### Nuitka

Use this when startup time and a compiled distribution become more important.

```powershell
.\packaging\build_nuitka.ps1
```

Nuitka builds can take much longer than PyInstaller builds.

## Notes

- Build from the project root.
- Keep source notebooks out of the packaged application unless they are explicitly needed.
- Test with real `.h5` and `.emd` files after every packaging change.
- If PyInstaller misses a dynamic py4DSTEM dependency, add it to `hiddenimports` in `py4dstem_pipeline.spec`.
- The scripts call tools through `python -m ...` where possible so they still work when user-site script folders are not on `PATH`.
