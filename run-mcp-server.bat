@echo off
echo ================================================================
echo          Obsidian-Milvus-FastMCP MCP Server
echo ================================================================
echo.
echo Current directory: %CD%
echo.

REM Check if mcp_server.py exists
if not exist "mcp_server.py" (
    echo ERROR: mcp_server.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

REM Check if config.py exists
if not exist "config.py" (
    echo ERROR: config.py not found in current directory
    echo Configuration file is required to run MCP server
    echo.
    pause
    exit /b 1
)

echo Starting MCP Server...
echo.
echo NOTE: This will start the MCP server for Claude Desktop
echo The server will run continuously until you stop it
echo Press Ctrl+C to stop the server
echo.

REM Run mcp_server.py with proper error handling
python mcp_server.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: MCP server exited with error code %errorlevel%
    echo Check the error messages above for details
    echo.
)

echo.
echo MCP Server stopped.
pause