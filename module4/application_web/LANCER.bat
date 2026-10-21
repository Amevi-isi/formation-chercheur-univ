@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo L'application n'est pas encore installee.
  echo Lancez d'abord INSTALLER_ET_LANCER.bat.
  pause
  exit /b 1
)

start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:8000"
".venv\Scripts\python.exe" run_app.py
