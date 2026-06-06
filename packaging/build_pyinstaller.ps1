param(
    [string]$EnvPath = ".conda\py4dstem-pipeline-packaging",
    [string]$SpecPath = "packaging\py4dstem_pipeline.spec",
    [string]$DistPath = "dist\pyinstaller",
    [switch]$SkipLaunchTest
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

    $runningApp = Get-Process -Name "py4DSTEM Pipeline" -ErrorAction SilentlyContinue
    if ($runningApp) {
        throw "Close the packaged py4DSTEM Pipeline application before rebuilding."
    }

    Write-Host "Building py4DSTEM Pipeline with PyInstaller in '$absoluteEnvPath'..."
    & $python -m PyInstaller --clean --noconfirm --distpath $DistPath $SpecPath
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE." }

    if (-not $SkipLaunchTest) {
        & "$PSScriptRoot\..\scripts\test_packaged_app.ps1" -ExePath "$DistPath\py4DSTEM Pipeline\py4DSTEM Pipeline.exe"
        if ($LASTEXITCODE -ne 0) { throw "Packaged application launch test failed with exit code $LASTEXITCODE." }
    }

    Write-Host "Build complete: $DistPath\py4DSTEM Pipeline"
}
finally {
    Pop-Location
}
