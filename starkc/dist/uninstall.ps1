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
$Current = Join-Path $InstallLib "current"
if (Test-Path $Current) {
    try {
        $Resolved = Resolve-Path $Current -ErrorAction Stop
        if ($Resolved.Path -like (Join-Path $InstallLib "versions\*")) {
            Remove-Item $Resolved.Path -Recurse -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
    Remove-Item $Current -Recurse -Force -ErrorAction SilentlyContinue
}
Remove-Item (Join-Path $InstallLib "versions") -Recurse -Force -ErrorAction SilentlyContinue
# Pre-versioned installations wrote their payload directly under `lib\stark` rather than into
# `versions\<v>`, so the removal above does not reach it. Named explicitly -- never a wildcard --
# because `$Prefix` is caller-supplied and `lib\stark` may hold what this installer did not write.
foreach ($Legacy in @(
    "stark-runtime",
    "stark-provider-abi",
    "starkc",
    "stark-env",
    "stark-file",
    "stark-net",
    "stark-random",
    "stark-time",
    "stark-tls"
)) {
    Remove-Item (Join-Path $InstallLib $Legacy) -Recurse -Force -ErrorAction SilentlyContinue
}
Remove-Item (Join-Path $InstallLib "uninstall.ps1") -Force -ErrorAction SilentlyContinue

if (-not $KeepPath) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Entries = @($UserPath -split ";" | Where-Object { $_ -and $_ -ne $InstallBin })
    [Environment]::SetEnvironmentVariable("Path", ($Entries -join ";"), "User")
}

Write-Host "Removed STARK from $Prefix"
