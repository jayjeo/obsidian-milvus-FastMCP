@echo off
chcp 65001 >nul
echo ================================================================
echo              FastMCP Server Simple Test
echo ================================================================
echo.

REM Change to the correct directory
g:
cd /d "G:\JJ Dropbox\J J\PythonWorks\milvus\obsidian-milvus-FastMCP"

echo Current directory: %CD%
echo.

echo [Step 1] Checking Python version...
python --version
echo.

echo [Step 2] Testing mcp module...
python test_mcp_module.py
echo.

echo [Step 3] Testing FastMCP module...
python test_fastmcp_module.py
echo.

echo [Step 4] Checking config file...
if exist "config.py" (
    echo [OK] config.py found
) else (
    echo [ERROR] config.py missing
    pause
    exit /b 1
)
echo.

echo [Step 5] Testing Milvus connection...
python test_milvus_connection.py
echo.

echo [Step 6] Starting FastMCP server...
echo ================================================================
echo NOTE: Server will start. Use Ctrl+C to stop.
echo ================================================================
echo.

timeout /t 3 /nobreak >nul

echo Starting FastMCP server...
python mcp_server.py

echo.
echo ================================================================
echo                Server Test Complete
echo ================================================================
pause
