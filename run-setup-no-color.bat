@echo off
echo ================================================================
echo       Milvus MCP Interactive Test and Setup Tool (No Colors)
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

echo Starting setup.py with colors disabled...
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

REM Run setup.py with colors disabled
set NO_COLOR=1
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
