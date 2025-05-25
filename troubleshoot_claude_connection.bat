@echo off
echo ============================================================
echo CLAUDE DESKTOP MCP CONNECTION TROUBLESHOOTER
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking Claude Desktop MCP server connection...
echo.

echo Step 1: Checking if MCP server is running...
echo --------------------------------------------------------
tasklist /fi "imagename eq python.exe" | findstr python >nul
if %errorlevel% equ 0 (
    echo ✅ Python processes are running
    tasklist /fi "imagename eq python.exe"
) else (
    echo ❌ No Python processes found
    echo MCP server may not be running
)

echo.
echo Step 2: Checking Claude Desktop configuration...
echo --------------------------------------------------------

set CLAUDE_CONFIG=%USERPROFILE%\AppData\Roaming\Claude\claude_desktop_config.json

if exist "%CLAUDE_CONFIG%" (
    echo ✅ Claude Desktop config found: %CLAUDE_CONFIG%
    echo.
    echo Configuration contents:
    echo ----------------------------------------
    type "%CLAUDE_CONFIG%"
    echo ----------------------------------------
) else (
    echo ❌ Claude Desktop config NOT found at: %CLAUDE_CONFIG%
    echo.
    echo This is likely the problem!
    echo Run create_claude_config.bat to create the configuration.
    goto end
)

echo.
echo Step 3: Checking configuration validity...
echo --------------------------------------------------------

REM Check if config contains obsidian-assistant
findstr "obsidian-assistant" "%CLAUDE_CONFIG%" >nul
if %errorlevel% equ 0 (
    echo ✅ obsidian-assistant server found in config
) else (
    echo ❌ obsidian-assistant server NOT found in config
    echo Configuration may be incomplete
)

REM Check if config contains mcp_server.py path
findstr "mcp_server.py" "%CLAUDE_CONFIG%" >nul
if %errorlevel% equ 0 (
    echo ✅ mcp_server.py path found in config
) else (
    echo ❌ mcp_server.py path NOT found in config
    echo Path configuration may be wrong
)

echo.
echo Step 4: Testing MCP server manually...
echo --------------------------------------------------------

if exist "start_mcp_verified.bat" (
    echo Testing verified MCP server startup...
    start /min "MCP Test" cmd /c "call start_mcp_verified.bat"
    timeout /t 3 /nobreak >nul
    echo ✅ MCP server test started
) else (
    echo ⚠️ No verified startup script found
    echo Try running fix_python_environment_complete.bat first
)

echo.
echo Step 5: Checking Claude Desktop process...
echo --------------------------------------------------------

tasklist /fi "imagename eq Claude.exe" | findstr Claude >nul
if %errorlevel% equ 0 (
    echo ✅ Claude Desktop is running
    echo.
    echo IMPORTANT: For configuration changes to take effect:
    echo 1. Close Claude Desktop completely
    echo 2. Wait 5 seconds
    echo 3. Restart Claude Desktop
    echo 4. Start a new conversation
) else (
    echo ❌ Claude Desktop is not running
    echo Please start Claude Desktop and try again
)

echo.
echo ============================================================
echo DIAGNOSIS SUMMARY
echo ============================================================
echo.

REM Quick diagnosis
if exist "%CLAUDE_CONFIG%" (
    if exist "start_mcp_verified.bat" (
        echo ✅ Configuration exists and MCP server setup found
        echo.
        echo LIKELY SOLUTION:
        echo 1. Restart Claude Desktop completely
        echo 2. Test MCP functionality in a new conversation
        echo.
        echo If still not working:
        echo 1. Run: create_claude_config.bat
        echo 2. Or run: claude_desktop_config_fixer.py
    ) else (
        echo ⚠️ Configuration exists but MCP server setup incomplete
        echo.
        echo SOLUTION:
        echo 1. Run: fix_python_environment_complete.bat
        echo 2. Then: create_claude_config.bat
    )
) else (
    echo ❌ No Claude Desktop configuration found
    echo.
    echo SOLUTION:
    echo 1. Run: create_claude_config.bat
    echo 2. Restart Claude Desktop
)

:end
echo.
pause
