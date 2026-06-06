param(
    [string]$EnvName = "4dstem",
    [string]$SpecPath = "packaging\py4dstem_pipeline.spec"
)

$ErrorActionPreference = "Stop"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    Write-Host "Building py4DSTEM Pipeline with PyInstaller in conda env '$EnvName'..."
    conda run -n $EnvName python -m pip install pyinstaller
    conda run -n $EnvName python -m PyInstaller --clean --noconfirm $SpecPath
    Write-Host "Build complete: dist\py4DSTEM Pipeline"
}
finally {
    Pop-Location
}
