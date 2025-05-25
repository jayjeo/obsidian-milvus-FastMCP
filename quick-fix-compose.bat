@echo off
echo Installing podman-compose to fix the compose issue...
echo.

:: Try different Python commands
python -m pip install podman-compose
if not errorlevel 1 goto success

py -m pip install podman-compose  
if not errorlevel 1 goto success

python3 -m pip install podman-compose
if not errorlevel 1 goto success

echo ERROR: Could not install podman-compose
echo Please ensure Python and pip are installed
pause
exit /b 1

:success
echo.
echo SUCCESS: podman-compose installed
echo You can now try running main.py again
echo.
pause
