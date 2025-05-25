@echo off
echo ============================================================
echo CLAUDE DESKTOP CONFIGURATION UPDATE
echo ============================================================
echo.

cd /d "%~dp0"
set PROJECT_DIR=%CD%

echo Current directory: %PROJECT_DIR%
echo.

echo Step 1: Finding Python executable...
echo --------------------------------------------------------

set WORKING_PYTHON=
for %%p in (python python3 py) do (
    %%p -c "import markdown; print('OK')" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('%%p -c "import sys; print(sys.executable)"') do (
            set WORKING_PYTHON=%%i
        )
        echo Found working Python: %%p
        goto found_python
    )
)

echo No working Python found with required modules
pause
exit /b 1

:found_python
echo Working Python: %WORKING_PYTHON%
echo.

echo Step 2: Creating Claude Desktop configuration...
echo --------------------------------------------------------

set CLAUDE_CONFIG_DIR=%USERPROFILE%\AppData\Roaming\Claude
set CLAUDE_CONFIG=%CLAUDE_CONFIG_DIR%\claude_desktop_config.json

if not exist "%CLAUDE_CONFIG_DIR%" (
    mkdir "%CLAUDE_CONFIG_DIR%"
)

if exist "%CLAUDE_CONFIG%" (
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" >nul
    echo Existing config backed up
)

REM Create configuration with encoding protection
(
echo {
echo   "mcpServers": {
echo     "obsidian-assistant": {
echo       "command": "%WORKING_PYTHON:\=\\%",
echo       "args": [
echo         "%PROJECT_DIR:\=\\%\\mcp_server.py"
echo       ],
echo       "env": {
echo         "PYTHONIOENCODING": "utf-8",
echo         "PYTHONLEGACYWINDOWSSTDIO": "utf-8",
echo         "PYTHONPATH": "%PROJECT_DIR:\=\\%"
echo       }
echo     }
echo   }
echo }
) > "%CLAUDE_CONFIG%"

echo Configuration created successfully!
echo.

echo Configuration file contents:
echo ============================================================
type "%CLAUDE_CONFIG%"
echo ============================================================

echo.
echo ============================================================
echo CONFIGURATION COMPLETED
echo ============================================================
echo.
echo Claude Desktop configuration has been updated with:
echo - Encoding protection (UTF-8)
echo - Correct Python path: %WORKING_PYTHON%
echo - Project path: %PROJECT_DIR%
echo.
echo NEXT STEPS:
echo.
echo 1. RESTART CLAUDE DESKTOP COMPLETELY
echo    - Close Claude Desktop entirely
echo    - Wait 5 seconds
echo    - Reopen Claude Desktop
echo.
echo 2. Start a new conversation in Claude Desktop
echo.
echo 3. Test by typing: "Search my notes for..."
echo.
echo If Claude Desktop shows "Server disconnected":
echo - The server is starting but may be slow due to high memory usage
echo - Wait a minute and try again
echo - The encoding issue should be fixed now
echo.

pause
