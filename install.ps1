# Color Grader: one-command install for Windows.
#   Right-click install.bat and run it, or:  powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Python packages..."
uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { throw "Package install failed." }

Write-Host "Checking ffmpeg..."
uv run python -m colorgrader.setup_ffmpeg
if ($LASTEXITCODE -ne 0) { throw "ffmpeg setup failed. See the message above." }

Write-Host ""
Write-Host "Done. Start the app with start.bat"
