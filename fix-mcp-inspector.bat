@echo off
chcp 65001 >nul
echo ================================================================
echo            MCP Inspector Fix Tool
echo ================================================================
echo.

echo [Step 1] Cleaning npm cache...
npm cache clean --force
if %errorlevel% neq 0 (
    echo WARNING: npm cache clean failed, continuing...
)
echo [OK] npm cache cleaned
echo.

echo [Step 2] Removing MCP CLI...
pip uninstall mcp -y
echo [OK] MCP CLI removed
echo.

echo [Step 3] Reinstalling MCP CLI...
pip install mcp
if %errorlevel% neq 0 (
    echo ERROR: MCP CLI installation failed
    pause
    exit /b 1
)
echo [OK] MCP CLI reinstalled
echo.

echo [Step 4] Checking Node.js version...
node --version
npm --version
echo.

echo ================================================================
echo                    Fix Complete!
echo ================================================================
echo Now try MCP Inspector again.
echo.

pause
