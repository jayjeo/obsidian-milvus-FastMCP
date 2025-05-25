@echo off
echo ================================================================
echo              Podman Auto-Startup Setup
echo ================================================================
echo.
echo Starting Python setup tool for Podman auto-startup...
echo This will use the integrated setup function from setup.py
echo.

REM Check if setup.py exists
if not exist "setup.py" (
    echo ERROR: setup.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

REM Run Python with specific menu option
echo Running: python -c "from setup import setup_podman_auto_startup; setup_podman_auto_startup()"
echo.

python -c "from setup import setup_podman_auto_startup; setup_podman_auto_startup()"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Auto-startup setup failed
    echo You can also use the standalone version: auto-startup-setup-standalone.bat
    echo.
) else (
    echo.
    echo Auto-startup setup completed successfully!
)

echo.
pause
