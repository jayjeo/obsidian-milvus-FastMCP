@echo off
setlocal

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ================================================================
echo       Milvus MCP Interactive Test and Setup Tool (Admin)
echo ================================================================
echo.
echo Current directory: %CD%
echo.

REM Check if setup.py exists
if not exist "setup.py" (
    echo ERROR: setup.py not found in current directory
    echo This is unexpected as we should be in the correct directory.
    echo.
    pause
    exit /b 1
)

echo Starting setup.py...
echo This tool will help you test and configure Milvus MCP system
echo.

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not found in PATH
    echo Please make sure Python is installed and in your PATH
    echo.
    pause
    exit /b 1
)

REM Run setup.py with proper error handling
echo Running: python setup.py
python setup.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: setup.py exited with error code %errorlevel%
    echo Please check if Python is properly installed and if setup.py has the correct permissions
    echo.
)

echo.
echo Setup tool finished.
pause
exit /b
