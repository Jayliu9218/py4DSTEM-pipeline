param(
    [string]$SourcePath = "dist\pyinstaller\py4DSTEM Pipeline",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    $versionFile = Join-Path (Get-Location) "app\version.py"
    $versionText = Get-Content $versionFile -Raw
    if ($versionText -notmatch '__version__\s*=\s*"([^"]+)"') {
        throw "Could not read __version__ from '$versionFile'."
    }
    $releaseTag = "v$($Matches[1])"
    if ([string]::IsNullOrWhiteSpace($OutputPath)) {
        $OutputPath = "dist\py4DSTEM-Pipeline-$releaseTag-portable.zip"
    }

    # The portable build is a zip of the PyInstaller onedir output, not a full
    # conda env. This keeps the archive small (only runtime files) and produces
    # a no-install portable distribution alongside the Inno Setup installer.
    $absoluteSource = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $SourcePath))
    if (-not (Test-Path (Join-Path $absoluteSource "py4DSTEM Pipeline.exe"))) {
        throw "PyInstaller build not found at '$absoluteSource'. Run packaging\build_pyinstaller.ps1 first."
    }

    Write-Host "Creating portable zip from '$absoluteSource'..."
    if (Test-Path $OutputPath) {
        Remove-Item $OutputPath -Force
    }
    Compress-Archive -Path (Join-Path $absoluteSource "*") -DestinationPath $OutputPath -CompressionLevel Optimal
    $sizeMb = [math]::Round((Get-Item $OutputPath).Length / 1MB, 1)
    Write-Host "Portable zip complete: $OutputPath ($sizeMb MB)"
}
finally {
    Pop-Location
}
