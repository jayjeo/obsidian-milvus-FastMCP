@echo off
echo ============================================================
echo MCP SERVER STARTUP - TIMEOUT PROTECTION MODE
echo ============================================================
echo.
echo This script addresses the infinite hanging issue by:
echo 1. Adding timeout protection
echo 2. Monitoring startup progress  
echo 3. Providing fallback options
echo 4. Preventing infinite waits
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Step 1: Running MCP startup diagnostics...
echo --------------------------------------------------------
echo This will test all components before starting the server
echo.

python mcp_startup_diagnostics.py
if %errorlevel% equ 0 (
    echo ✅ Diagnostics passed - proceeding with startup
) else (
    echo ❌ Diagnostics failed - checking alternatives
    goto alternatives
)

echo.
echo Step 2: Starting MCP server with timeout protection...
echo --------------------------------------------------------
echo Using timeout-protected VBS launcher
echo Maximum runtime: 2 minutes
echo.

start "MCP Server" auto_start_mcp_server_timeout.vbs

echo ✅ MCP Server startup initiated with timeout protection
echo.
echo Monitoring logs for 30 seconds...
timeout /t 30 /nobreak >nul

if exist "auto_startup_mcp.log" (
    echo.
    echo Latest log entries:
    echo --------------------------------------------------------
    powershell -Command "Get-Content 'auto_startup_mcp.log' | Select-Object -Last 10"
) else (
    echo ⚠️ Log file not found yet - server may still be starting
)

echo.
echo ============================================================
echo STARTUP COMPLETED
echo ============================================================
echo.
echo The MCP server has been started with timeout protection.
echo.
echo To monitor progress:
echo   - Check auto_startup_mcp.log
echo   - Check mcp_startup_debug.log  
echo   - Check vbs_startup.log
echo.
echo If the server is stuck, it will automatically timeout and exit.
echo.
goto end

:alternatives
echo.
echo ============================================================
echo ALTERNATIVE STARTUP METHODS
echo ============================================================
echo.
echo Diagnostics failed. Try these alternatives:
echo.
echo 1. Direct MCP server startup:
echo    python mcp_server.py
echo.
echo 2. Interactive mode:
echo    python main.py
echo.
echo 3. Manual diagnostics:
echo    python check_python_env.py
echo.
echo 4. Environment setup:
echo    setup_environment.bat
echo.

:end
pause
