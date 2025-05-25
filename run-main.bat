@echo off
echo ================================================================
echo          Obsidian-Milvus-FastMCP Main Program
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

echo Starting main.py...
echo.

REM Run main.py with proper error handling
python main.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: main.py exited with error code %errorlevel%
    echo.
)

echo.
echo Program finished.
pause