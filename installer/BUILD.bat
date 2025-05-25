@echo off
echo Building Obsidian-Milvus-FastMCP Installer...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Run the advanced build script
echo Running advanced build script...
python build_advanced.py

if %errorlevel% equ 0 (
    echo.
    echo Build completed successfully!
    echo Check the 'dist' folder for the installer
) else (
    echo.
    echo Build failed! Check the error messages above
)

pause
