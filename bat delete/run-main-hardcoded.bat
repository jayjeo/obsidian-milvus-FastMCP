@echo off
setlocal enabledelayedexpansion

REM Hardcoded Path Version of run-main.bat
REM This version uses absolute paths instead of relative paths

echo ================================================================
echo          Obsidian-Milvus-FastMCP Main Program (Hardcoded)
echo ================================================================
echo.

REM Set hardcoded project path
set "PROJECT_PATH=G:\JJ Dropbox\J J\PythonWorks\milvus\obsidian-milvus-FastMCP"
set "MAIN_SCRIPT=%PROJECT_PATH%\main.py"

echo Project Path: !PROJECT_PATH!
echo Main Script: !MAIN_SCRIPT!
echo Current Directory: %CD%
echo.

REM Check if project directory exists
if not exist "!PROJECT_PATH!" (
    echo ERROR: Project directory not found: !PROJECT_PATH!
    echo Please update the PROJECT_PATH variable in this script
    echo.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "!MAIN_SCRIPT!" (
    echo ERROR: main.py not found at: !MAIN_SCRIPT!
    echo Please verify the file exists in the project directory
    echo.
    pause
    exit /b 1
)

REM Change to project directory
echo Changing to project directory...
cd /d "!PROJECT_PATH!"
if %errorlevel% neq 0 (
    echo ERROR: Failed to change to project directory
    echo.
    pause
    exit /b 1
)

echo Successfully changed to: %CD%
echo.

REM Find Python executable
echo Searching for Python executable...
set "PYTHON_CMD="

REM Try python command first
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
    echo Found Python: python
    goto :python_found
)

REM Try python3 command
python3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python3"
    echo Found Python: python3
    goto :python_found
)

REM Try py command
py --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=py"
    echo Found Python: py
    goto :python_found
)

REM Try specific paths
set "PYTHON_PATHS[0]=C:\Python\python.exe"
set "PYTHON_PATHS[1]=C:\Python311\python.exe"
set "PYTHON_PATHS[2]=C:\Python312\python.exe"
set "PYTHON_PATHS[3]=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
set "PYTHON_PATHS[4]=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHON_PATHS[5]=C:\Users\%USERNAME%\anaconda3\python.exe"
set "PYTHON_PATHS[6]=C:\Users\%USERNAME%\miniconda3\python.exe"

for /l %%i in (0,1,6) do (
    if exist "!PYTHON_PATHS[%%i]!" (
        "!PYTHON_PATHS[%%i]!" --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=!PYTHON_PATHS[%%i]!"
            echo Found Python at: !PYTHON_PATHS[%%i]!
            goto :python_found
        )
    )
)

echo ERROR: Python not found in any common locations
echo Please install Python or update the Python paths in this script
echo.
pause
exit /b 1

:python_found
echo.
echo Starting main.py with !PYTHON_CMD!...
echo Command: !PYTHON_CMD! "!MAIN_SCRIPT!"
echo Working Directory: %CD%
echo.

REM Run main.py with proper error handling
"!PYTHON_CMD!" "!MAIN_SCRIPT!"
set "EXIT_CODE=%errorlevel%"

echo.
if !EXIT_CODE! equ 0 (
    echo Success: Program completed successfully
) else (
    echo Error: Program exited with error code !EXIT_CODE!
    echo.
    echo Common solutions:
    echo - Check if all required packages are installed: pip install -r requirements.txt
    echo - Verify Milvus is running
    echo - Check config.py settings
    echo - Review the error messages above
)

echo.
echo Program finished at %DATE% %TIME%
pause
