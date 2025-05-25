@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo         MCP Server Auto-Startup Uninstaller
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

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

echo This will remove the auto-startup configuration for the MCP server.
echo.
echo The following will be removed:
echo   - Windows scheduled task: MainPyOption1AutoStartup
echo   - Manual control scripts (optional)
echo   - Log files (optional)
echo.
echo NOTE: auto_startup_main_option1.vbs will NOT be removed (core file)
echo.

set /p "CONFIRM=Do you want to continue? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

echo.
echo ================================================================
echo                  Removing Auto-Startup
echo ================================================================
echo.

REM Stop any running MCP server processes first
echo Stopping any running MCP server processes...
taskkill /f /im python.exe /fi "WINDOWTITLE eq *main.py*" 2>nul
taskkill /f /im python3.exe /fi "WINDOWTITLE eq *main.py*" 2>nul
taskkill /f /im py.exe /fi "WINDOWTITLE eq *main.py*" 2>nul
echo [OK] Processes stopped

echo.
echo Removing scheduled task...
schtasks /query /tn "MainPyOption1AutoStartup" >nul 2>&1
if !errorlevel! equ 0 (
    schtasks /delete /tn "MainPyOption1AutoStartup" /f >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Scheduled task removed successfully
    ) else (
        echo [ERROR] Failed to remove scheduled task
        echo You may need to remove it manually from Task Scheduler
    )
) else (
    echo [INFO] Scheduled task was not found (already removed)
)

echo.
echo Removing scheduled task...
set /p "REMOVE_LOGS=Do you want to remove log files? (y/N): "
if /i "%REMOVE_LOGS%"=="y" (
    echo Removing log files...
    
    if exist "auto_startup_main.log" (
        del "auto_startup_main.log" >nul 2>&1
        if !errorlevel! equ 0 (
            echo [OK] Removed auto_startup_main.log
        ) else (
            echo [ERROR] Could not remove auto_startup_main.log
        )
    )
    
    if exist "temp_input.txt" (
        del "temp_input.txt" >nul 2>&1
        echo [OK] Cleaned up temporary files
    )
    
    if exist "temp_start_option1.bat" (
        del "temp_start_option1.bat" >nul 2>&1
        echo [OK] Cleaned up temporary files
    )
) else (
    echo [INFO] Log files kept for reference
)

echo.
echo ================================================================
echo                  Uninstall Complete
echo ================================================================
echo.

echo The MCP server auto-startup has been removed.
echo.
echo What was removed:
echo   - Windows scheduled task: MainPyOption1AutoStartup
echo   - Manual control scripts (kept for manual use)
if /i "%REMOVE_LOGS%"=="y" (
    echo   - Log files
) else (
    echo   - Log files (kept)
)
echo.

echo The MCP server will no longer start automatically with Windows.
echo.
echo If you want to start the MCP server manually in the future:
echo   1. Open Command Prompt in the project directory
echo   2. Run: python main.py
echo   3. Select option 1 (Start MCP Server)
echo.

echo Uninstall completed successfully!
pause
