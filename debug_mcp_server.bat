@echo off
echo Starting MCP Server with logging...
echo.

REM Kill any existing Python processes
echo Killing existing Python processes...
taskkill /F /IM python.exe 2>nul

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start MCP server with output logging using PowerShell
echo Starting MCP server...
powershell -Command "& {python 'D:\obsidian-milvus-FastMCP\mcp_server.py' 2>&1 | Tee-Object -FilePath 'mcp_server_output.log'}"

pause
