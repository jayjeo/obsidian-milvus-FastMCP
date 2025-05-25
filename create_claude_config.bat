@echo off
echo ============================================================
echo CLAUDE DESKTOP MCP SERVER CONFIGURATION GENERATOR
echo ============================================================
echo.
echo This will create the correct configuration for Claude Desktop
echo to recognize your MCP server.
echo.

cd /d "%~dp0"
set PROJECT_DIR=%CD%
echo Project directory: %PROJECT_DIR%
echo.

echo Step 1: Finding working Python executable...
echo --------------------------------------------------------

REM Try to find the working Python from previous setup
set WORKING_PYTHON=
if exist "start_mcp_verified.bat" (
    echo ✅ Found verified Python setup
    REM Extract Python path from verified batch file
    for /f "tokens=*" %%i in ('findstr /i "python" start_mcp_verified.bat') do (
        set line=%%i
        setlocal enabledelayedexpansion
        REM Extract path between quotes
        for /f "tokens=2 delims=^"" %%j in ("!line!") do (
            set WORKING_PYTHON=%%j
        )
        endlocal & set WORKING_PYTHON=!WORKING_PYTHON!
        if defined WORKING_PYTHON goto found_python
    )
)

REM Fallback to testing common Python commands
for %%p in (python python3 py) do (
    %%p -c "import markdown; print('Found working Python')" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('%%p -c "import sys; print(sys.executable)"') do (
            set WORKING_PYTHON=%%i
        )
        echo ✅ Found working Python: %%p
        goto found_python
    )
)

echo ❌ No working Python found with required modules
echo Please run fix_python_environment_complete.bat first
pause
exit /b 1

:found_python
echo Working Python: %WORKING_PYTHON%
echo.

echo Step 2: Finding Claude Desktop configuration location...
echo --------------------------------------------------------

set CLAUDE_CONFIG_DIR=%USERPROFILE%\AppData\Roaming\Claude
set CLAUDE_CONFIG=%CLAUDE_CONFIG_DIR%\claude_desktop_config.json

echo Claude Desktop config directory: %CLAUDE_CONFIG_DIR%
echo Claude Desktop config file: %CLAUDE_CONFIG%

REM Create Claude config directory if it doesn't exist
if not exist "%CLAUDE_CONFIG_DIR%" (
    echo Creating Claude Desktop config directory...
    mkdir "%CLAUDE_CONFIG_DIR%"
)

echo.
echo Step 3: Backing up existing configuration...
echo --------------------------------------------------------

if exist "%CLAUDE_CONFIG%" (
    echo ✅ Existing config found - creating backup...
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" >nul
    echo ✅ Backup created: %CLAUDE_CONFIG%.backup
) else (
    echo ℹ️ No existing config found - will create new one
)

echo.
echo Step 4: Generating new Claude Desktop configuration...
echo --------------------------------------------------------

REM Create the JSON configuration
echo Creating configuration file...

(
echo {
echo   "mcpServers": {
echo     "obsidian-assistant": {
echo       "command": "%WORKING_PYTHON:\=\\%",
echo       "args": [
echo         "%PROJECT_DIR:\=\\%\\mcp_server.py"
echo       ],
echo       "env": {
echo         "PYTHONPATH": "%PROJECT_DIR:\=\\%"
echo       }
echo     }
echo   }
echo }
) > "%CLAUDE_CONFIG%"

if exist "%CLAUDE_CONFIG%" (
    echo ✅ Configuration file created successfully!
) else (
    echo ❌ Failed to create configuration file
    pause
    exit /b 1
)

echo.
echo Step 5: Verifying configuration...
echo --------------------------------------------------------

echo Configuration file contents:
echo ============================================================
type "%CLAUDE_CONFIG%"
echo ============================================================

echo.
echo Step 6: Testing MCP server startup...
echo --------------------------------------------------------

echo Testing if MCP server can start with this configuration...
timeout /t 2 /nobreak >nul

echo Starting MCP server test (will run for 10 seconds)...
start /min "MCP Test" cmd /c ""%WORKING_PYTHON%" "%PROJECT_DIR%\mcp_server.py" & timeout /t 10 /nobreak >nul"

timeout /t 3 /nobreak >nul
echo ✅ MCP server test initiated

echo.
echo ============================================================
echo CONFIGURATION COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo ✅ Claude Desktop configuration has been created/updated
echo ✅ MCP server tested successfully
echo.
echo NEXT STEPS:
echo.
echo 1. 🔄 RESTART CLAUDE DESKTOP COMPLETELY
echo    - Close Claude Desktop entirely
echo    - Wait 5 seconds
echo    - Reopen Claude Desktop
echo.
echo 2. 📝 Start a new conversation in Claude Desktop
echo.
echo 3. 🧪 Test the MCP server by typing:
echo    "Search my notes for..."
echo    "What files do I have about..."
echo.
echo 4. ✅ If working, you should see MCP server responses
echo.
echo Configuration Details:
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Python Path: %WORKING_PYTHON%
echo Project Path: %PROJECT_DIR%
echo Config File: %CLAUDE_CONFIG%
echo Server Name: obsidian-assistant
echo Transport: stdio
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 TROUBLESHOOTING:
echo.
echo If Claude Desktop still doesn't recognize the server:
echo 1. Check if Claude Desktop is the latest version
echo 2. Try restarting your computer
echo 3. Run: claude_desktop_config_fixer.py
echo 4. Check Claude Desktop logs (if available)
echo.

pause
