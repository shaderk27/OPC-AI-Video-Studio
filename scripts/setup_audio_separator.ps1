param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "models\audio-separator"
$virtualEnv = Join-Path $runtimeRoot ".venv"
$pythonExe = Join-Path $virtualEnv "Scripts\python.exe"
$separatorExe = Join-Path $virtualEnv "Scripts\audio-separator.exe"

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $PythonLauncher "-$PythonVersion" -m venv $virtualEnv
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --retries 5 --timeout 60 "audio-separator[cpu]==0.44.5"
if ($LASTEXITCODE -ne 0) {
    throw "audio-separator dependency installation failed; rerun when the package index connection is stable"
}

if (-not (Test-Path -LiteralPath $separatorExe)) {
    throw "audio-separator was installed but its executable was not found at $separatorExe"
}

Write-Host "Audio separator runtime ready: $separatorExe"
Write-Host "The first localization run downloads UVR_MDXNET_KARA_2.onnx into $runtimeRoot."
