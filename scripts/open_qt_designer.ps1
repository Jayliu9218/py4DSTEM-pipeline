param(
    [string]$UiFile = "",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

function Invoke-DesignerPath {
    param(
        [string]$Path,
        [string[]]$Arguments
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    if ($CheckOnly) {
        Write-Host "Qt Designer found: $Path"
        return $true
    }

    if ($Arguments.Count -gt 0) {
        & $Path @Arguments
    } else {
        & $Path
    }
    return $true
}

function Invoke-DesignerCommand {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $resolved) {
        return $false
    }

    return Invoke-DesignerPath -Path $resolved.Source -Arguments $Arguments
}

function Get-PySideDesignerPath {
    $probe = @"
from pathlib import Path
try:
    import PySide6
except Exception:
    raise SystemExit(1)

base = Path(PySide6.__file__).resolve().parent
for candidate in [
    base / "designer.exe",
    base / "Qt" / "bin" / "designer.exe",
]:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
"@
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $result = python -c $probe 2>$null
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($LASTEXITCODE -eq 0 -and $result) {
        return [string]$result
    }
    return ""
}

$designerArgs = @()
if ($UiFile) {
    $designerArgs += $UiFile
}

if (Invoke-DesignerCommand -Command "pyside6-designer" -Arguments $designerArgs) {
    exit 0
}

if (Invoke-DesignerCommand -Command "designer" -Arguments $designerArgs) {
    exit 0
}

$pySideDesignerPath = Get-PySideDesignerPath
if ($pySideDesignerPath -and (Invoke-DesignerPath -Path $pySideDesignerPath -Arguments $designerArgs)) {
    exit 0
}

$userPythonDesignerPath = Join-Path $env:APPDATA "Python\Python313\site-packages\PySide6\designer.exe"
if (Invoke-DesignerPath -Path $userPythonDesignerPath -Arguments $designerArgs) {
    exit 0
}

Write-Error @"
Qt Designer was not found in the current environment.

Activate the development environment first:
  conda activate 4dstem

Then run:
  .\scripts\open_qt_designer.ps1

If Designer is still unavailable, reinstall PySide6 in that environment:
  python -m pip install PySide6

You can also edit scripts\open_qt_designer.ps1 and add another Designer path if
your PySide6 installation is stored somewhere else.
"@
