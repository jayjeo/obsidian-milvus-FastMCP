@echo off
setlocal

echo ================================================================
echo       Milvus MCP Auto-Startup Setup (Administrator Required)
echo ================================================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script requires administrator privileges.
    echo Please right-click on this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Check if setup.py exists
if not exist "setup.py" (
    echo ERROR: setup.py not found in directory: %CD%
    echo This is unexpected as we should be in the correct directory.
    echo.
    pause
    exit /b 1
)

echo Running setup.py with option 0 (Setup Podman Auto-Startup)...
echo.

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not found in PATH
    echo Please make sure Python is installed and in your PATH
    echo.
    pause
    exit /b 1
)

REM Run setup.py with auto-selection of option 0
set NO_COLOR=1
echo import sys; sys.argv = ["setup.py", "0"]; exec(open("setup.py").read()) > run_option_0.py
python run_option_0.py
del run_option_0.py

echo.
echo Auto-startup setup completed.
pause
exit /b
