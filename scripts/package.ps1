# Packs the built executables, their runtime DLLs (FFmpeg, curl, zlib, dav1d, OpenSSL from vcpkg) and
# serve.ps1 into dist\ninfer-windows.zip. cudart is linked statically, so the machine that runs the
# zip needs only the NVIDIA driver. Run scripts\build.ps1 first. Attach the zip to a GitHub release by hand.
param([string]$Out = "$(Split-Path $PSScriptRoot -Parent)\dist\ninfer-windows.zip")
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$apps = "$root\build\turing-windows\apps"
$exes = 'ninfer.exe', 'ninfer-serve.exe', 'ninfer-perplexity.exe'
foreach ($exe in $exes) {
    if (-not (Test-Path "$apps\$exe")) { Write-Host "error: $apps\$exe not found. Run scripts\build.ps1 first." -ForegroundColor Red; exit 1 }
}

$stage = Join-Path $env:TEMP "ninfer-package-$PID"
New-Item -ItemType Directory $stage | Out-Null
try {
    Copy-Item $exes.ForEach({ "$apps\$_" }) $stage
    Copy-Item "$apps\*.dll" $stage          # vcpkg copies the runtime DLLs next to the executables
    Copy-Item "$PSScriptRoot\serve.ps1", "$root\LICENSE" $stage
    New-Item -ItemType Directory (Split-Path $Out) -Force | Out-Null
    Remove-Item $Out -ErrorAction SilentlyContinue
    Compress-Archive -Path "$stage\*" -DestinationPath $Out
} finally {
    Remove-Item $stage -Recurse -Force
}
Write-Host ("Wrote {0} ({1:N0} MB)" -f $Out, ((Get-Item $Out).Length / 1MB))
