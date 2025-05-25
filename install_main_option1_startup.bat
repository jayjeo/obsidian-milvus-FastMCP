@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo        Main.py Option 1 Auto-Startup Installer
echo        (Starts MCP Server automatically on Windows boot)
echo ================================================================
echo.

REM Check for administrator rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script requires administrator privileges.
    echo.
    echo Please right-click on this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo [OK] Running with administrator privileges
echo.

REM Get script directory and force change to it
set "SCRIPT_DIR=%~dp0"
echo Script directory: %SCRIPT_DIR%

REM Force change to script directory (fix for admin execution)
cd /d "%SCRIPT_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Cannot change to script directory: %SCRIPT_DIR%
    echo This usually happens when running as administrator.
    pause
    exit /b 1
)

echo Current directory: %CD%
echo Script directory matches current: %SCRIPT_DIR%
echo.

REM Check if required files exist
if not exist "auto_startup_main_option1.vbs" (
    echo ERROR: auto_startup_main_option1.vbs not found in current directory.
    echo Please make sure all files are in the same directory.
    pause
    exit /b 1
)

if not exist "main.py" (
    echo ERROR: main.py not found in current directory.
    echo Please make sure this installer is in the project root directory.
    pause
    exit /b 1
)

if not exist "config.py" (
    echo ERROR: config.py not found in current directory.
    echo Please make sure this installer is in the project root directory.
    pause
    exit /b 1
)

echo [OK] Required files found
echo.

REM Test Python installation
echo Checking Python installation...

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python found in PATH
    set "PYTHON_CMD=python"
    goto :test_config
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python3 found in PATH
    set "PYTHON_CMD=python3"
    goto :test_config
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python found via py launcher
    set "PYTHON_CMD=py"
    goto :test_config
)

echo ERROR: Python not found.
echo.
echo Please install Python first:
echo   1. Download from https://www.python.org/
echo   2. Make sure to check "Add Python to PATH" during installation
echo   3. Restart your computer after installation
echo.
pause
exit /b 1

:test_config
echo Testing project configuration...

REM Test if config.py can be imported
%PYTHON_CMD% -c "import config; print('Config loaded successfully')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Could not load config.py
    echo Please check if all required Python packages are installed.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Configuration is valid
echo.

REM Test Podman configuration
echo Checking Podman configuration...
%PYTHON_CMD% -c "from config import get_podman_path; print(get_podman_path())" >temp_podman_test.txt 2>&1
if %errorlevel% equ 0 (
    set /p PODMAN_PATH=<temp_podman_test.txt
    del temp_podman_test.txt >nul 2>&1
    echo [OK] Podman configured at: !PODMAN_PATH!
) else (
    if exist temp_podman_test.txt del temp_podman_test.txt >nul 2>&1
    echo Warning: Podman configuration might have issues
    echo The auto-startup will still be installed but may not work properly
)
echo.

REM Check if task already exists
echo Checking for existing auto-startup task...
schtasks /query /tn "MainPyOption1AutoStartup" >nul 2>&1
if %errorlevel% equ 0 (
    echo Found existing task. Removing it first...
    schtasks /delete /tn "MainPyOption1AutoStartup" /f >nul 2>&1
    if %errorlevel% neq 0 (
        echo Warning: Could not remove existing task
    ) else (
        echo [OK] Existing task removed
    )
)

echo.
echo ================================================================
echo          Creating Windows Startup Task for MCP Server
echo ================================================================

REM Create the scheduled task
set "VBS_PATH=%SCRIPT_DIR%auto_startup_main_option1.vbs"

echo Creating scheduled task...
echo Task name: MainPyOption1AutoStartup
echo Script path: %VBS_PATH%
echo Action: Start MCP Server (main.py option 1)
echo.

schtasks /create /tn "MainPyOption1AutoStartup" /tr "wscript.exe \"%VBS_PATH%\"" /sc onstart /ru SYSTEM /rl highest /f /delay 0002:00 >nul 2>&1

REM Verify task was created
schtasks /query /tn "MainPyOption1AutoStartup" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Scheduled task created successfully
) else (
    echo ERROR: Failed to create scheduled task.
    pause
    exit /b 1
)

echo.

REM Create manual start/stop scripts for convenience
echo Creating convenience scripts...

REM Manual start script
set "MANUAL_START=%SCRIPT_DIR%manual_start_mcp_server.bat"
(
echo @echo off
echo echo Starting MCP Server manually...
echo cd /d "%SCRIPT_DIR%"
echo echo 1 ^| %PYTHON_CMD% main.py
) > "%MANUAL_START%"

REM Manual stop script  
set "MANUAL_STOP=%SCRIPT_DIR%manual_stop_mcp_server.bat"
(
echo @echo off
echo echo Stopping MCP Server...
echo echo This will close all Python processes running main.py
echo pause
echo taskkill /f /im python.exe /fi "WINDOWTITLE eq main.py*" 2^>nul
echo taskkill /f /im python3.exe /fi "WINDOWTITLE eq main.py*" 2^>nul
echo taskkill /f /im py.exe /fi "WINDOWTITLE eq main.py*" 2^>nul
echo echo MCP Server stopped
echo pause
) > "%MANUAL_STOP%"

echo [OK] Convenience scripts created
echo   - %MANUAL_START%
echo   - %MANUAL_STOP%
echo.

echo ================================================================
echo              Installation Complete!
echo ================================================================
echo.
echo Installation finished successfully!
pause
