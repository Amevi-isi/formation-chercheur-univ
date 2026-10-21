@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo Installation du Laboratoire IA...
echo.

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto :erreur
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :erreur

call LANCER.bat
exit /b 0

:erreur
echo.
echo L'installation n'a pas abouti. Verifiez que Python 3 et Internet sont disponibles.
pause
exit /b 1

