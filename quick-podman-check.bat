@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo           Quick Podman Diagnostic
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo ================================================================
echo System Information
echo ================================================================

echo Windows version:
ver
echo.

echo WSL status:
wsl --status 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] WSL command failed or not installed
    echo This might be the root cause of Podman issues
) else (
    echo [OK] WSL is available
)
echo.

echo ================================================================
echo Podman Information
echo ================================================================

REM Find Podman
set "PODMAN_CMD="
if exist "C:\Program Files\RedHat\Podman\podman.exe" (
    set "PODMAN_CMD=C:\Program Files\RedHat\Podman\podman.exe"
    echo [OK] Found Podman: %PODMAN_CMD%
) else (
    for /f "tokens=*" %%p in ('where podman 2^>nul') do (
        set "PODMAN_CMD=%%p"
        echo [OK] Found Podman in PATH: !PODMAN_CMD!
        goto :found_podman
    )
    echo [ERROR] Podman not found
    goto :no_podman
)

:found_podman

echo.
echo Podman version:
"%PODMAN_CMD%" --version
echo.

echo Podman machine list:
"%PODMAN_CMD%" machine list
echo.

echo Podman system connection:
"%PODMAN_CMD%" system connection list
echo.

echo ================================================================
echo Connection Test
echo ================================================================

echo Testing Podman connection...
"%PODMAN_CMD%" ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] ✅ Podman connection working
    
    echo.
    echo Current containers:
    "%PODMAN_CMD%" ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo.
    echo Testing Milvus specifically...
    "%PODMAN_CMD%" ps --format "{{.Names}}" | findstr milvus >nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] ✅ Milvus containers found
        
        echo Testing Milvus API connection...
        python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('[SUCCESS] ✅ Milvus API responding'); connections.disconnect('default')" 2>nul
        if %errorlevel% equ 0 (
            echo [SUCCESS] ✅ Everything is working! You can run main.py
        ) else (
            echo [INFO] ⚠️ Milvus containers exist but API not responding yet
            echo This is normal if containers just started - wait 2-3 minutes
        )
    ) else (
        echo [INFO] ⚠️ No Milvus containers found
        echo Run: start-milvus.bat to start Milvus services
    )
    
) else (
    echo [ERROR] ❌ Podman connection failed
    echo.
    echo Error details:
    "%PODMAN_CMD%" ps
    echo.
    echo Common causes:
    echo 1. Podman machine not running
    echo 2. WSL2 issues
    echo 3. Socket connection problems
    echo.
    echo Solutions to try:
    echo 1. Run: complete-podman-reset.bat
    echo 2. Restart computer
    echo 3. Check WSL2 installation
)

goto :end

:no_podman
echo.
echo Podman is not installed or not in PATH.
echo.
echo Please install Podman:
echo 1. Download from: https://podman.io/
echo 2. Or use winget: winget install RedHat.Podman
echo 3. Restart your computer after installation
echo.

:end
echo.
echo ================================================================
echo Quick Action Guide
echo ================================================================
echo.

if defined PODMAN_CMD (
    "%PODMAN_CMD%" ps >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ System is ready! You can:
        echo   • Run: run-main.bat
        echo   • Or if no Milvus containers: start-milvus.bat
    ) else (
        echo ❌ Podman connection issues detected:
        echo   • Run: complete-podman-reset.bat (recommended)
        echo   • Or manually: podman machine stop, then podman machine start
    )
) else (
    echo ❌ Podman not found:
    echo   • Install Podman from: https://podman.io/
    echo   • Or use: winget install RedHat.Podman
)

echo.
pause
