@echo off
echo ============================================================
echo CLAUDE DESKTOP CONFIG UPDATE WITH ENCODING FIX
echo ============================================================
echo.

cd /d "%~dp0"
set PROJECT_DIR=%CD%

echo Step 1: Setting up encoding environment...
echo --------------------------------------------------------
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo ✅ Encoding environment configured
echo.

echo Step 2: Finding working Python with encoding support...
echo --------------------------------------------------------

set WORKING_PYTHON=
if exist "start_mcp_verified.bat" (
    echo Found verified Python setup
    for /f "tokens=*" %%i in ('findstr /i "python" start_mcp_verified.bat') do (
        for /f "tokens=2 delims=^"" %%j in ("%%i") do (
            set WORKING_PYTHON=%%j
            goto found_python
        )
    )
)

REM Fallback
for %%p in (python python3 py) do (
    %%p -c "import markdown; print('Python with modules OK')" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('%%p -c "import sys; print(sys.executable)"') do (
            set WORKING_PYTHON=%%i
        )
        echo ✅ Found working Python: %%p
        goto found_python
    )
)

echo ❌ No working Python found with required modules
pause
exit /b 1

:found_python
echo Working Python: %WORKING_PYTHON%
echo.

echo Step 3: Creating encoding-safe startup script...
echo --------------------------------------------------------

REM Create an encoding-safe startup script
echo @echo off > start_mcp_safe.bat
echo chcp 65001 ^>nul 2^>^&1 >> start_mcp_safe.bat
echo set PYTHONIOENCODING=utf-8 >> start_mcp_safe.bat
echo set PYTHONLEGACYWINDOWSSTDIO=utf-8 >> start_mcp_safe.bat
echo cd /d "%PROJECT_DIR%" >> start_mcp_safe.bat
echo "%WORKING_PYTHON%" mcp_server.py >> start_mcp_safe.bat

echo ✅ Safe startup script created: start_mcp_safe.bat
echo.

echo Step 4: Updating Claude Desktop configuration...
echo --------------------------------------------------------

set CLAUDE_CONFIG_DIR=%USERPROFILE%\AppData\Roaming\Claude
set CLAUDE_CONFIG=%CLAUDE_CONFIG_DIR%\claude_desktop_config.json

if not exist "%CLAUDE_CONFIG_DIR%" (
    mkdir "%CLAUDE_CONFIG_DIR%"
)

if exist "%CLAUDE_CONFIG%" (
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" >nul
    echo ✅ Existing config backed up
)

REM Create new configuration with encoding-safe startup
echo Creating Claude Desktop configuration...

(
echo {
echo   "mcpServers": {
echo     "obsidian-assistant": {
echo       "command": "cmd",
echo       "args": [
echo         "/c",
echo         "cd /d \"%PROJECT_DIR%\" && start_mcp_safe.bat"
echo       ],
echo       "env": {
echo         "PYTHONIOENCODING": "utf-8",
echo         "PYTHONLEGACYWINDOWSSTDIO": "utf-8",
echo         "PYTHONPATH": "%PROJECT_DIR%"
echo       }
echo     }
echo   }
echo }
) > "%CLAUDE_CONFIG%"

echo ✅ Configuration updated with encoding protection
echo.

echo Step 5: Testing the configuration...
echo --------------------------------------------------------

echo Configuration file contents:
echo ============================================================
type "%CLAUDE_CONFIG%"
echo ============================================================

echo.
echo Step 6: Testing MCP server startup...
echo --------------------------------------------------------

echo Testing MCP server with new configuration...
echo (This will run for 10 seconds then stop)

start /min "MCP Test" cmd /c "call start_mcp_safe.bat & timeout /t 10 /nobreak >nul"
timeout /t 5 /nobreak >nul

echo ✅ MCP server test initiated
echo.

echo ============================================================
echo CONFIGURATION COMPLETED WITH ENCODING FIX
echo ============================================================
echo.
echo ✅ Claude Desktop configuration updated with encoding protection
echo ✅ Safe startup script created (start_mcp_safe.bat)
echo ✅ Unicode/emoji issues should be resolved
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
echo.
echo Configuration Details:
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Python Path: %WORKING_PYTHON%
echo Project Path: %PROJECT_DIR%
echo Config File: %CLAUDE_CONFIG%
echo Safe Startup: start_mcp_safe.bat
echo Encoding: UTF-8 with fallback protection
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 TROUBLESHOOTING:
echo.
echo If you still see encoding errors:
echo 1. Run: start_mcp_with_encoding_fix.bat
echo 2. Check Windows regional settings
echo 3. Update Python to latest version
echo 4. Restart computer after configuration
echo.

pause
