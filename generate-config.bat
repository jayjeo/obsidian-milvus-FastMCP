@echo off
setlocal enabledelayedexpansion
echo ================================================================
echo         Auto-generating config.json for Podman paths
echo ================================================================

REM Check for Podman in common locations
set PODMAN_FOUND=0

REM Check system installation
if exist "C:\Program Files\RedHat\Podman\podman.exe" (
    set PODMAN_PATH=C:\\Program Files\\RedHat\\Podman\\podman.exe
    set PODMAN_FOUND=1
    echo ✅ Found Podman: C:\Program Files\RedHat\Podman\podman.exe
)

REM Check user installation  
if exist "%USERPROFILE%\AppData\Local\Programs\RedHat\Podman\podman.exe" (
    set PODMAN_PATH=%USERPROFILE%\\AppData\\Local\\Programs\\RedHat\\Podman\\podman.exe
    set PODMAN_FOUND=1
    echo ✅ Found Podman: %USERPROFILE%\AppData\Local\Programs\RedHat\Podman\podman.exe
)

REM Check PATH
if !PODMAN_FOUND! equ 0 (
    where podman >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('where podman') do (
            set PODMAN_PATH_RAW=%%i
            set PODMAN_PATH=!PODMAN_PATH_RAW:\=\\!
            set PODMAN_FOUND=1
            echo ✅ Found Podman in PATH: %%i
        )
    )
)

if !PODMAN_FOUND! equ 0 (
    echo ❌ Podman not found! Please install Podman first.
    echo    Install with: winget install RedHat.Podman
    pause
    exit /b 1
)

echo.
echo Generating config.json...
echo {
echo   "podman_path": "!PODMAN_PATH!",
echo   "compose_command": "podman compose"
echo } > config.json

echo.
echo ✅ config.json created successfully!
echo.
echo Contents:
type config.json

echo.
echo ================================================================
echo Copy this config.json to your application directory
echo ================================================================
pause