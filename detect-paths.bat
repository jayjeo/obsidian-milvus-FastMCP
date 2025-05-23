@echo off
setlocal enabledelayedexpansion
title Conda/Mamba Path Detection Utility

echo ================================================================
echo      Conda/Mamba Path Detection Utility
echo ================================================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python first
    pause
    exit /b 1
)

echo Python found - running path detection...
echo.

if not exist "detect-paths.py" (
    echo ERROR: detect-paths.py not found
    echo Please ensure detect-paths.py is in the same directory
    pause
    exit /b 1
)

REM Run the Python script for detection
python detect-paths.py

if %errorlevel% neq 0 (
    echo.
    echo Error occurred while running path detection
    pause
    exit /b 1
)

echo.
echo Path detection completed successfully!
echo.

REM Check if config-snippet was generated
if exist "config-snippet.txt" (
    echo Configuration snippet saved to: config-snippet.txt
    echo.
    set /p view="Would you like to view the generated config snippet? (y/n): "
    if /i "!view!"=="y" (
        echo.
        echo ----------------------------------------
        type config-snippet.txt
        echo ----------------------------------------
        echo.
    )
    
    set /p copy="Would you like to copy this to config.py? (y/n): "
    if /i "!copy!"=="y" (
        if exist "config.py" (
            echo Creating backup of existing config.py...
            copy config.py config.py.backup >nul
            echo Backup created: config.py.backup
        )
        
        echo Updating config.py with detected settings...
        copy config.py + config-snippet.txt config-temp.py >nul 2>&1
        if exist "config-temp.py" (
            move config-temp.py config.py >nul
            echo Configuration updated successfully!
        ) else (
            copy config-snippet.txt config.py >nul
            echo Configuration created successfully!
        )
    )
)

echo.
echo What to do next:
echo    1. Review the configuration recommendations above
echo    2. Edit config.py if you need custom paths
echo    3. Run one-click-install.bat to test installation
echo.

set /p run="Would you like to run the installation now? (y/n): "
if /i "!run!"=="y" (
    if exist "one-click-install.bat" (
        echo.
        echo Starting installation...
        call one-click-install.bat
    ) else (
        echo one-click-install.bat not found
    )
)

pause
