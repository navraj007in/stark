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

if (-not (Test-Path (Join-Path $PackageBin "stark.exe") -PathType Leaf) -or
    -not (Test-Path (Join-Path $PackageRuntime "Cargo.toml") -PathType Leaf)) {
    throw "install.ps1 must be run from an extracted STARK release package"
}

$InstallBin = Join-Path $Prefix "bin"
$InstallLib = Join-Path $Prefix "lib\stark"
$InstallRuntime = Join-Path $InstallLib "stark-runtime"
New-Item -ItemType Directory -Force -Path $InstallBin, $InstallLib | Out-Null

Copy-Item (Join-Path $PackageBin "stark.exe") $InstallBin -Force
Copy-Item (Join-Path $PackageBin "starkc.exe") $InstallBin -Force
Copy-Item (Join-Path $PackageBin "starkide.exe") $InstallBin -Force
if (Test-Path $InstallRuntime) {
    Remove-Item $InstallRuntime -Recurse -Force
}
Copy-Item $PackageRuntime $InstallRuntime -Recurse
Copy-Item (Join-Path $PackageDir "uninstall.ps1") $InstallLib -Force

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
Write-Host "Run: $InstallBin\stark.exe --help"
