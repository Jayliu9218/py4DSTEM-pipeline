param(
    [string]$EnvPath = ".conda\py4dstem-pipeline-packaging",
    [string]$BaseEnvName = "4dstem",
    [string]$Py4DSTEMRef = "dev"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    $absoluteEnvPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $EnvPath))

    if (-not (Test-Path (Join-Path $absoluteEnvPath "python.exe"))) {
        Write-Host "Cloning runtime dependencies from conda env '$BaseEnvName' into '$absoluteEnvPath'..."
        conda create -p $absoluteEnvPath --clone $BaseEnvName -y
        if ($LASTEXITCODE -ne 0) { throw "Packaging environment clone failed with exit code $LASTEXITCODE." }
    }

    $python = Join-Path $absoluteEnvPath "python.exe"
    Write-Host "Installing project runtime dependencies..."
    & $python -m pip install --upgrade -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed with exit code $LASTEXITCODE." }

    Write-Host "Installing packaging-only tools..."
    & $python -m pip install --upgrade -r requirements.packaging.txt
    if ($LASTEXITCODE -ne 0) { throw "Packaging tool installation failed with exit code $LASTEXITCODE." }

    Write-Host "Installing py4DSTEM from GitHub ref '$Py4DSTEMRef'..."
    & $python -m pip install --upgrade --upgrade-strategy only-if-needed "py4DSTEM @ git+https://github.com/py4dstem/py4DSTEM.git@$Py4DSTEMRef"
    if ($LASTEXITCODE -ne 0) { throw "py4DSTEM Git installation failed with exit code $LASTEXITCODE." }

    conda env config vars set -p $absoluteEnvPath PYTHONNOUSERSITE=1 PYTHONUTF8=1
    if ($LASTEXITCODE -ne 0) { throw "Packaging environment isolation setup failed with exit code $LASTEXITCODE." }

    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency check failed with exit code $LASTEXITCODE." }

    & $python "$PSScriptRoot\check_packaging_env.py" --expected-ref $Py4DSTEMRef
    if ($LASTEXITCODE -ne 0) { throw "Packaging environment source check failed with exit code $LASTEXITCODE." }

    Write-Host "Packaging environment ready: $absoluteEnvPath"
}
finally {
    Pop-Location
}
