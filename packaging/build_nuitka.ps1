param(
    [string]$EnvPath = ".conda\py4dstem-pipeline-packaging"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    $absoluteEnvPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $EnvPath))
    $python = Join-Path $absoluteEnvPath "python.exe"
    if (-not (Test-Path $python)) {
        throw "Packaging environment not found. Run scripts\setup_packaging_env.ps1 first."
    }

    Write-Host "Building py4DSTEM Pipeline with Nuitka in '$absoluteEnvPath'..."
    # --include-package list mirrors the PyInstaller spec's collect_all set so
    # the two backends bundle the same dependency surface. py4DSTEM's __init__
    # eagerly imports io/visualize/process/..., so its hard transitive deps
    # (scikit-*, ncempy, hdf5plugin, gdown, pylops, colorspacious, pymatgen +
    # monty/spglib) must be included explicitly.
    & $python -m nuitka main.py `
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
        --include-package=scipy `
        --include-package=scikit-image `
        --include-package=scikit-learn `
        --include-package=scikit-optimize `
        --include-package=ncempy `
        --include-package=hdf5plugin `
        --include-package=gdown `
        --include-package=pylops `
        --include-package=colorspacious `
        --include-package=pymatgen `
        --include-package=monty `
        --include-package=spglib `
        --nofollow-import-to=dask `
        --nofollow-import-to=distributed `
        --nofollow-import-to=mpire `
        --nofollow-import-to=plotly `
        --nofollow-import-to=pandas `
        --nofollow-import-to=sympy `
        --nofollow-import-to=notebook `
        --nofollow-import-to=IPython `
        --nofollow-import-to=pytest `
        --output-dir=dist\nuitka
    if ($LASTEXITCODE -ne 0) { throw "Nuitka build failed with exit code $LASTEXITCODE." }
    Write-Host "Nuitka build complete: dist\nuitka"
}
finally {
    Pop-Location
}
