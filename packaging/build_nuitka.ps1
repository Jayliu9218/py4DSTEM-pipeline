param(
    [string]$EnvName = "4dstem"
)

$ErrorActionPreference = "Stop"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    Write-Host "Building py4DSTEM Pipeline with Nuitka in conda env '$EnvName'..."
    conda run -n $EnvName python -m pip install nuitka ordered-set zstandard
    conda run -n $EnvName python -m nuitka main.py `
        --standalone `
        --windows-console-mode=disable `
        --enable-plugin=pyside6 `
        --include-package=app `
        --include-package=py4DSTEM `
        --include-package=emdfile `
        --include-package=pyqtgraph `
        --include-package=h5py `
        --include-package=tifffile `
        --include-package=matplotlib `
        --output-dir=dist\nuitka
    Write-Host "Nuitka build complete: dist\nuitka"
}
finally {
    Pop-Location
}
