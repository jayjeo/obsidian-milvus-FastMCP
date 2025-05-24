@echo off
chcp 65001 >nul
echo =======================================================
echo MCP Server Auto Start Setup Tool - Universal Version
echo =======================================================
echo.

REM Get current directory as project path
set "PROJECT_PATH=%~dp0"
REM Remove trailing backslash
if "%PROJECT_PATH:~-1%"=="\" set "PROJECT_PATH=%PROJECT_PATH:~0,-1%"

set "VBS_FILE=%PROJECT_PATH%\auto_start_mcp_server.vbs"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_NAME=MCP Server Auto Start"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\%SHORTCUT_NAME%.lnk"

echo Project folder (auto-detected): %PROJECT_PATH%
echo.

echo Checking VBS file...
if not exist "%VBS_FILE%" (
    echo ERROR: auto_start_mcp_server.vbs file not found in current directory!
    echo Expected location: %VBS_FILE%
    echo.
    echo Please make sure this batch file is in the same folder as:
    echo - auto_start_mcp_server.vbs
    echo - mcp_server.py
    pause
    exit /b 1
)
echo SUCCESS: VBS file found: auto_start_mcp_server.vbs
echo.

echo Checking mcp_server.py...
if not exist "%PROJECT_PATH%\mcp_server.py" (
    echo WARNING: mcp_server.py not found in current directory!
    echo The VBS script will fail when executed.
    echo Please make sure mcp_server.py is in the same folder.
    echo.
    echo Continue anyway? (Y/N)
    set /p "CONTINUE="
    if /i not "%CONTINUE%"=="Y" exit /b 1
)
echo SUCCESS: mcp_server.py found
echo.

echo Startup programs folder: %STARTUP_FOLDER%
echo.

echo Registering to Windows startup programs...

REM Remove existing shortcut if it exists
if exist "%SHORTCUT_PATH%" (
    echo Removing existing shortcut...
    del "%SHORTCUT_PATH%" >nul 2>&1
)

REM Create new shortcut using PowerShell
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%VBS_FILE%'; $Shortcut.WorkingDirectory = '%PROJECT_PATH%'; $Shortcut.Description = 'MCP Server Auto Start - Universal'; $Shortcut.Save()}"

if exist "%SHORTCUT_PATH%" (
    echo.
    echo ===============================================
    echo SUCCESS: Auto-start setup completed!
    echo ===============================================
    echo.
    echo Configuration:
    echo - Project folder: %PROJECT_PATH%
    echo - VBS script: auto_start_mcp_server.vbs
    echo - Shortcut created: %SHORTCUT_NAME%.lnk
    echo - Execution mode: Background without window
    echo - Notification: Shows when server starts/fails
    echo.
    echo The MCP server will now start automatically when:
    echo - Windows starts up
    echo - User logs in
    echo - Computer restarts
    echo.
    echo To test immediately, run: auto_start_mcp_server.vbs
    echo To remove auto-start, delete: %SHORTCUT_PATH%
    echo.
    echo SHARING INSTRUCTIONS:
    echo 1. Copy entire project folder to target computer
    echo 2. Run setup_auto_start.bat on target computer
    echo 3. Auto-start will be configured automatically
) else (
    echo.
    echo ===============================================
    echo ERROR: Startup registration failed!
    echo ===============================================
    echo.
    echo Possible solutions:
    echo 1. Run this batch file as Administrator
    echo 2. Check Windows security settings
    echo 3. Manually create shortcut in startup folder:
    echo    Target: %VBS_FILE%
    echo    Folder: %STARTUP_FOLDER%
)

echo.
echo Press any key to exit...
pause >nul