param(
    [string]$EnvPath = ".conda\py4dstem-pipeline-packaging",
    [string]$OutputPath = "dist\py4DSTEM-Pipeline-conda-pack.zip"
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

    Write-Host "Packing conda env '$absoluteEnvPath' with conda-pack..."
    & $python -m conda_pack -p $absoluteEnvPath -o $OutputPath --force
    if ($LASTEXITCODE -ne 0) { throw "Conda-pack failed with exit code $LASTEXITCODE." }
    Write-Host "Conda-pack archive complete: $OutputPath"
}
finally {
    Pop-Location
}
