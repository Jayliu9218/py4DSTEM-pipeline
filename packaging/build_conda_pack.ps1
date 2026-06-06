param(
    [string]$EnvName = "4dstem",
    [string]$OutputPath = "dist\py4DSTEM-Pipeline-conda-pack.zip"
)

$ErrorActionPreference = "Stop"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    Write-Host "Packing conda env '$EnvName' with conda-pack..."
    conda run -n $EnvName python -m pip install conda-pack
    conda run -n $EnvName python -c "import sys; from conda_pack.cli import main; sys.exit(main())" -n $EnvName -o $OutputPath --force
    Write-Host "Conda-pack archive complete: $OutputPath"
}
finally {
    Pop-Location
}
