param(
    [string]$Prefix = $(if ($env:STARK_INSTALL_PREFIX) {
        $env:STARK_INSTALL_PREFIX
    } else {
        Join-Path $env:LOCALAPPDATA "Programs\STARK"
    }),
    [switch]$NoPathUpdate
)

$ErrorActionPreference = "Stop"
$PackageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageBin = Join-Path $PackageDir "bin"
$PackageRuntime = Join-Path $PackageDir "lib\stark\stark-runtime"
$PackageProviderAbi = Join-Path $PackageDir "lib\stark\stark-provider-abi"
$PackageManifest = Join-Path $PackageDir "manifest.json"

if (-not (Test-Path (Join-Path $PackageBin "stark.exe") -PathType Leaf) -or
    -not (Test-Path $PackageManifest -PathType Leaf) -or
    -not (Test-Path (Join-Path $PackageRuntime "Cargo.toml") -PathType Leaf) -or
    -not (Test-Path (Join-Path $PackageProviderAbi "Cargo.toml") -PathType Leaf)) {
    throw "install.ps1 must be run from an extracted STARK release package"
}

$Manifest = Get-Content $PackageManifest -Raw | ConvertFrom-Json
$Version = $Manifest.stark_version
if (-not $Version) {
    throw "manifest.json does not declare stark_version"
}

$InstallBin = Join-Path $Prefix "bin"
$InstallLib = Join-Path $Prefix "lib\stark"
$VersionsRoot = Join-Path $InstallLib "versions"
$VersionRoot = Join-Path $VersionsRoot $Version
$StagingRoot = Join-Path $InstallLib ".staging-$Version-$PID"
New-Item -ItemType Directory -Force -Path $InstallBin, $VersionsRoot | Out-Null
if (Test-Path $StagingRoot) {
    Remove-Item $StagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null

Copy-Item (Join-Path $PackageDir "bin") (Join-Path $StagingRoot "bin") -Recurse
Copy-Item (Join-Path $PackageDir "lib") (Join-Path $StagingRoot "lib") -Recurse
Copy-Item $PackageManifest $StagingRoot -Force
Copy-Item (Join-Path $PackageDir "BUILD-INFO.txt") $StagingRoot -Force
Copy-Item (Join-Path $PackageDir "LICENSE") $StagingRoot -Force
Copy-Item (Join-Path $PackageDir "README.md") $StagingRoot -Force
Copy-Item (Join-Path $PackageDir "install.ps1") $StagingRoot -Force
Copy-Item (Join-Path $PackageDir "uninstall.ps1") $StagingRoot -Force

& (Join-Path $StagingRoot "bin\stark.exe") doctor --root $StagingRoot | Out-Null
if ($LASTEXITCODE -ne 0) {
    Remove-Item $StagingRoot -Recurse -Force
    throw "staged STARK installation failed manifest verification"
}

if (Test-Path $VersionRoot) {
    Remove-Item $VersionRoot -Recurse -Force
}
Move-Item $StagingRoot $VersionRoot
$Current = Join-Path $InstallLib "current"
if (Test-Path $Current) {
    Remove-Item $Current -Recurse -Force
}
try {
    New-Item -ItemType Junction -Path $Current -Target $VersionRoot | Out-Null
} catch {
    Copy-Item $VersionRoot $Current -Recurse
}

Copy-Item (Join-Path $Current "bin\stark.exe") $InstallBin -Force
Copy-Item (Join-Path $Current "bin\starkc.exe") $InstallBin -Force
Copy-Item (Join-Path $Current "bin\starkide.exe") $InstallBin -Force
Copy-Item (Join-Path $Current "uninstall.ps1") $InstallLib -Force

if (-not $NoPathUpdate) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Entries = @($UserPath -split ";" | Where-Object { $_ })
    if ($Entries -notcontains $InstallBin) {
        $NewPath = (@($Entries) + $InstallBin) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        $env:Path = "$env:Path;$InstallBin"
        Write-Host "Added $InstallBin to the user PATH."
    }
}

Write-Host "Installed STARK in $Prefix"
Write-Host "Version: $Version"
Write-Host "Run: $InstallBin\stark.exe --help"
