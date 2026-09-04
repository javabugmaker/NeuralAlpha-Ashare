@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run scripts\setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_gui.py
if errorlevel 1 pause
