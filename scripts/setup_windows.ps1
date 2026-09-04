$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev,fast]"
& .\.venv\Scripts\alpha-ashare.exe doctor
Write-Host "Setup complete. Run run_gui.bat or activate .venv and use alpha-ashare."
