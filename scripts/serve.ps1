# Starts ninfer-serve with the default flags. Without -Model it uses models\qwen3_8_27b.ninfer and
# downloads it from Hugging Face (neroued/Qwen3.8-27B-NInfer, about 18 GB) when that file is missing.
# Extra ninfer-serve arguments: scripts\serve.ps1 -ServerArgs '--max-context','32768'
param([string]$Model, [int]$Port = 8080, [string[]]$ServerArgs)
$ErrorActionPreference = 'Stop'
# Inside the release zip serve.ps1 sits beside the executables; in the repository it is in scripts\.
$packaged = Test-Path "$PSScriptRoot\ninfer-serve.exe"
$root = if ($packaged) { $PSScriptRoot } else { Split-Path $PSScriptRoot -Parent }
function Fail($message) { Write-Host "error: $message" -ForegroundColor Red; exit 1 }

$exe = @("$root\build\windows\apps\ninfer-serve.exe", "$PSScriptRoot\ninfer-serve.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { Fail 'ninfer-serve.exe was not found. Run scripts\build.ps1 first.' }

if (-not $Model) {
    $Model = "$root\models\qwen3_8_27b.ninfer"
    if (-not (Test-Path $Model)) {
        $python = (Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if (-not $python) { Fail "$Model is missing and Python was not found to download it. Install Python 3 or pass -Model PATH." }
        Write-Host 'Downloading neroued/Qwen3.8-27B-NInfer (about 18 GB)...'
        & $python -m pip install --quiet --upgrade huggingface_hub
        & $python -c "from huggingface_hub import hf_hub_download; hf_hub_download('neroued/Qwen3.8-27B-NInfer', 'qwen3_8_27b.ninfer', local_dir=r'$root\models')"
        if (-not (Test-Path $Model)) { Fail 'the model download failed.' }
    }
}
if (-not (Test-Path $Model)) { Fail "model not found: $Model" }

Write-Host "Serving at http://127.0.0.1:$Port/v1  (Ctrl+C to stop)"
& $exe $Model --port $Port @ServerArgs
exit $LASTEXITCODE
