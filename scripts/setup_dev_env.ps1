param(
    [ValidateSet("Check", "Create", "Install", "Validate", "Test", "All")]
    [string]$Stage = "All",
    [string]$EnvPath = ".conda\py4dstem-pipeline-dev",
    [string]$PythonVersion = "3.11",
    [string]$Py4DSTEMRef = "dev",
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-LastExitCode([string]$Action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE."
    }
}

function Assert-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

function Test-Prerequisites {
    Write-Step "Checking machine prerequisites"
    Assert-Command "conda" "Install Miniconda or Anaconda, then reopen PowerShell."
    Assert-Command "git" "Install Git for Windows, then reopen PowerShell."
    Write-Host "Conda and Git are available."
}

function New-DevelopmentEnvironment([string]$AbsoluteEnvPath) {
    Write-Step "Creating the development environment"
    $python = Join-Path $AbsoluteEnvPath "python.exe"
    if (Test-Path $python) {
        Write-Host "Environment already exists: $AbsoluteEnvPath"
        return
    }
    conda create -p $AbsoluteEnvPath "python=$PythonVersion" pip -y
    Assert-LastExitCode "Conda environment creation"
}

function Assert-EnvironmentExists([string]$AbsoluteEnvPath) {
    $python = Join-Path $AbsoluteEnvPath "python.exe"
    if (-not (Test-Path $python)) {
        throw "Development environment not found. Run this script with -Stage Create or -Stage All first."
    }
    return $python
}

function Install-DevelopmentDependencies([string]$AbsoluteEnvPath) {
    Write-Step "Installing or refreshing project dependencies"
    $python = Assert-EnvironmentExists $AbsoluteEnvPath
    & $python -m pip install --upgrade pip setuptools wheel
    Assert-LastExitCode "pip bootstrap"
    & $python -m pip install --upgrade -r requirements.txt
    Assert-LastExitCode "Project dependency installation"
    & $python -m pip install --upgrade --upgrade-strategy only-if-needed "py4DSTEM @ git+https://github.com/py4dstem/py4DSTEM.git@$Py4DSTEMRef"
    Assert-LastExitCode "py4DSTEM installation"
    conda env config vars set -p $AbsoluteEnvPath PYTHONNOUSERSITE=1 PYTHONUTF8=1
    Assert-LastExitCode "Environment isolation configuration"
}

function Test-DevelopmentEnvironment([string]$AbsoluteEnvPath) {
    Write-Step "Validating the development environment"
    $python = Assert-EnvironmentExists $AbsoluteEnvPath
    & $python -m pip check
    Assert-LastExitCode "pip dependency check"
    & $python scripts\check_runtime_dependencies.py
    Assert-LastExitCode "Runtime dependency check"
    & $python scripts\check_packaging_env.py --expected-ref $Py4DSTEMRef
    Assert-LastExitCode "py4DSTEM source check"
}

function Invoke-ProjectTests([string]$AbsoluteEnvPath) {
    Write-Step "Running project tests and compile checks"
    $python = Assert-EnvironmentExists $AbsoluteEnvPath
    & $python -m unittest discover -s tests
    Assert-LastExitCode "Unit tests"
    & $python -m compileall -q app tests main.py
    Assert-LastExitCode "Compile check"
}

Push-Location (Resolve-Path "$PSScriptRoot\..")
try {
    $absoluteEnvPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $EnvPath))
    Test-Prerequisites

    switch ($Stage) {
        "Check" { }
        "Create" { New-DevelopmentEnvironment $absoluteEnvPath }
        "Install" { Install-DevelopmentDependencies $absoluteEnvPath }
        "Validate" { Test-DevelopmentEnvironment $absoluteEnvPath }
        "Test" { Invoke-ProjectTests $absoluteEnvPath }
        "All" {
            New-DevelopmentEnvironment $absoluteEnvPath
            Install-DevelopmentDependencies $absoluteEnvPath
            Test-DevelopmentEnvironment $absoluteEnvPath
        }
    }

    $python = Join-Path $absoluteEnvPath "python.exe"
    Write-Host ""
    if (Test-Path $python) {
        Write-Host "Development environment ready: $absoluteEnvPath" -ForegroundColor Green
        Write-Host "Run the application: & `"$python`" .\main.py"
        Write-Host "Run full verification: .\scripts\setup_dev_env.ps1 -Stage Test"
        Write-Host "Refresh dependencies later: .\scripts\setup_dev_env.ps1 -Stage Install"
    }
    else {
        Write-Host "Prerequisite check complete. Create the environment with:" -ForegroundColor Green
        Write-Host ".\scripts\setup_dev_env.ps1 -Stage All"
    }

    if ($Launch) {
        if (-not (Test-Path $python)) {
            throw "Cannot launch because the development environment has not been created."
        }
        Write-Step "Launching py4DSTEM Pipeline"
        Start-Process -FilePath $python -ArgumentList ".\main.py"
    }
}
finally {
    Pop-Location
}
