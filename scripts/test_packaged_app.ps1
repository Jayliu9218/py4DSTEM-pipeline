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
