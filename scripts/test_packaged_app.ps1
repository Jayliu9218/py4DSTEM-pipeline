param(
    [string]$ExePath = "dist\pyinstaller\py4DSTEM Pipeline\py4DSTEM Pipeline.exe",
    [int]$WaitSeconds = 8
)

$ErrorActionPreference = "Stop"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    $absoluteExePath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $ExePath))
    if (-not (Test-Path $absoluteExePath)) {
        throw "Packaged application not found. Run packaging\build_pyinstaller.ps1 first."
    }

    $bundleRoot = Split-Path $absoluteExePath -Parent
    $resourceRoot = Join-Path $bundleRoot "_internal"
    foreach ($relativePath in @(
        "app\theme.qss",
        "app\theme_light.qss",
        "images\py4DSTEM_logo.png"
    )) {
        $resourcePath = Join-Path $resourceRoot $relativePath
        if (-not (Test-Path $resourcePath)) {
            throw "Packaged visual resource is missing: $resourcePath"
        }
    }

    $process = Start-Process -FilePath $absoluteExePath -PassThru
    Start-Sleep -Seconds $WaitSeconds

    if ($process.HasExited) {
        throw "Packaged application exited early with code $($process.ExitCode)."
    }

    Stop-Process -Id $process.Id -Force
    Write-Host "Packaged application launch test passed."
}
finally {
    Pop-Location
}
