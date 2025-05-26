@echo off
echo ================================================================
echo    Obsidian-Milvus-FastMCP Installer Build Script
echo ================================================================
echo.
echo This script will build the complete installer executable.
echo.

REM Change to the installer directory
cd /d "%~dp0"

echo Current directory: %CD%
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

echo Python found. Checking for required files...
echo.

REM Check if required files exist
if not exist "installer_complete.py" (
    echo ERROR: installer_complete.py not found
    echo Please make sure all installer files are present
    echo.
    pause
    exit /b 1
)

if not exist "build_advanced.py" (
    echo ERROR: build_advanced.py not found
    echo Please make sure all installer files are present
    echo.
    pause
    exit /b 1
)

echo ✓ All required files found
echo.
echo Starting installer build process...
echo This may take several minutes...
echo.

REM Run the advanced build script
python build_advanced.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build process failed with error code %errorlevel%
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                    BUILD COMPLETED!
echo ================================================================
echo.
echo The installer has been built successfully.
echo.
echo You can find the installer at:
echo   dist/ObsidianMilvusInstaller.exe
echo.
echo For distribution, use the release package:
echo   release/ObsidianMilvusInstaller.exe
echo   release/README.md
echo.
echo Users should:
echo 1. Install Anaconda first (https://www.anaconda.com/download)
echo 2. Run ObsidianMilvusInstaller.exe as Administrator
echo 3. Follow the installation wizard
echo.
echo ================================================================
echo.
pause
