@echo off
echo ================================================================
echo       Milvus MCP Interactive Test and Setup Tool
echo ================================================================
echo.
echo Current directory: %CD%
echo.

REM Check if setup.py exists
if not exist "setup.py" (
    echo ERROR: setup.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

echo Starting setup.py...
echo This tool will help you test and configure Milvus MCP system
echo.

REM Run setup.py with proper error handling
python setup.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: setup.py exited with error code %errorlevel%
    echo.
)

echo.
echo Setup tool finished.
pause