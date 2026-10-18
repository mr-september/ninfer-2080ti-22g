# Builds ninfer, ninfer-serve and ninfer-perplexity on native Windows (Turing, sm_75).
# Prerequisites: NVIDIA driver, CUDA 12.8+, Visual Studio 2022 or Build Tools with "Desktop
# development with C++". The first run installs FFmpeg and curl through vcpkg.json (about 20 minutes).
param([switch]$Dev)   # -Dev also builds the tests and benchmarks
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
function Fail($message) { Write-Host "error: $message" -ForegroundColor Red; exit 1 }

# 1-2. Find Visual Studio and enter its x64 developer shell.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vs) { Fail 'Visual Studio 2022 (or Build Tools) with "Desktop development with C++" was not found. Install it from https://visualstudio.microsoft.com/downloads/' }
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation 2>&1 | Out-Null

# 3. CUDA 12.8+, and a host compiler that nvcc accepts (a quick probe compile).
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { Fail 'nvcc was not found. Install the CUDA Toolkit 12.8 or newer from https://developer.nvidia.com/cuda-downloads' }
$cuda = [version](nvcc --version | Select-String 'release (\d+\.\d+)').Matches[0].Groups[1].Value
if ($cuda -lt [version]'12.8') { Fail "CUDA $cuda is too old; install 12.8 or newer." }
$probe = Join-Path $env:TEMP 'ninfer_nvcc_probe.cu'
'__global__ void k() {}' | Set-Content $probe
$null = nvcc -arch=sm_75 -c $probe -o "$probe.obj" 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail "nvcc $cuda rejects this MSVC toolset (cl $((cl 2>&1 | Select-Object -First 1) -replace '.*Version ','')). Install an older toolset in the Visual Studio Installer ('MSVC v143 - VS 2022 C++ x64/x86 build tools', an earlier 14.4x) or a newer CUDA Toolkit."
}

# vcpkg: VCPKG_ROOT, else the copy bundled with Visual Studio, else a local clone in .vcpkg.
if (-not $env:VCPKG_ROOT) {
    $bundled = "$vs\VC\vcpkg"
    $env:VCPKG_ROOT = if (Test-Path "$bundled\vcpkg.exe") { $bundled } else { "$root\.vcpkg" }
}
if (-not (Test-Path "$env:VCPKG_ROOT\vcpkg.exe")) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'git is needed to fetch vcpkg; install Git for Windows or set VCPKG_ROOT to a vcpkg checkout.' }
    if (-not (Test-Path "$env:VCPKG_ROOT\bootstrap-vcpkg.bat")) { git clone https://github.com/microsoft/vcpkg $env:VCPKG_ROOT }
    & "$env:VCPKG_ROOT\bootstrap-vcpkg.bat" -disableMetrics
}

# 4. Configure and build.
Set-Location $root
$preset = if ($Dev) { 'turing-windows-dev' } else { 'turing-windows' }
cmake --preset $preset; if ($LASTEXITCODE) { Fail 'configure failed' }
cmake --build --preset $preset; if ($LASTEXITCODE) { Fail 'build failed' }

# 5. Result.
Write-Host "`nBuilt: $root\build\turing-windows\apps\ninfer.exe, ninfer-serve.exe, ninfer-perplexity.exe"
Write-Host 'Next: scripts\serve.ps1'
