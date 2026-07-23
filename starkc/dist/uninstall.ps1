param(
    [string]$Prefix = $(if ($env:STARK_INSTALL_PREFIX) {
        $env:STARK_INSTALL_PREFIX
    } else {
        Join-Path $env:LOCALAPPDATA "Programs\STARK"
    }),
    [switch]$KeepPath
)

$ErrorActionPreference = "Stop"
$InstallBin = Join-Path $Prefix "bin"
$InstallLib = Join-Path $Prefix "lib\stark"

Remove-Item (Join-Path $InstallBin "stark.exe") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $InstallBin "starkc.exe") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $InstallBin "starkide.exe") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $InstallLib "stark-runtime") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $InstallLib "uninstall.ps1") -Force -ErrorAction SilentlyContinue

if (-not $KeepPath) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Entries = @($UserPath -split ";" | Where-Object { $_ -and $_ -ne $InstallBin })
    [Environment]::SetEnvironmentVariable("Path", ($Entries -join ";"), "User")
}

Write-Host "Removed STARK from $Prefix"
