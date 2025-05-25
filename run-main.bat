@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo          Obsidian-Milvus-FastMCP Main Program
echo          (Smart Python Environment Detection)
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

echo ================================================================
echo Finding correct Python environment...
echo ================================================================

REM Check execution mode
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running as ADMINISTRATOR
    echo Searching for Python with required packages...
) else (
    echo [INFO] Running as NORMAL USER
)
echo.

set "PYTHON_FOUND=0"
set "PYTHON_CMD="

REM Method 1: Try current Python in PATH
echo Testing default Python...
python -c "import markdown; import pymilvus" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    set "PYTHON_FOUND=1"
    echo [SUCCESS] Default Python has required packages
    goto :run_program
) else (
    echo [INFO] Default Python missing packages
)

REM Method 2: Try python3
echo Testing python3...
python3 -c "import markdown; import pymilvus" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python3"
    set "PYTHON_FOUND=1"
    echo [SUCCESS] python3 has required packages
    goto :run_program
) else (
    echo [INFO] python3 missing packages or not found
)

REM Method 3: Try py launcher
echo Testing py launcher...
py -c "import markdown; import pymilvus" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
    set "PYTHON_FOUND=1"
    echo [SUCCESS] py launcher has required packages
    goto :run_program
) else (
    echo [INFO] py launcher missing packages or not found
)

REM Method 4: Search common Python installation paths
echo Searching common Python installation paths...

set "PYTHON_PATHS[0]=%USERPROFILE%\AppData\Local\Programs\Python"
set "PYTHON_PATHS[1]=%LOCALAPPDATA%\Programs\Python"
set "PYTHON_PATHS[2]=C:\Python"
set "PYTHON_PATHS[3]=C:\Program Files\Python"
set "PYTHON_PATHS[4]=C:\Program Files (x86)\Python"
set "PYTHON_PATHS[5]=%PROGRAMFILES%\Python"

for /L %%i in (0,1,5) do (
    if defined PYTHON_PATHS[%%i] (
        set "BASE_PATH=!PYTHON_PATHS[%%i]!"
        if exist "!BASE_PATH!" (
            echo Checking: !BASE_PATH!
            
            REM Look for Python subdirectories
            for /d %%d in ("!BASE_PATH!\Python*") do (
                set "PYTHON_EXE=%%d\python.exe"
                if exist "!PYTHON_EXE!" (
                    echo Testing: !PYTHON_EXE!
                    "!PYTHON_EXE!" -c "import markdown; import pymilvus" >nul 2>&1
                    if !errorlevel! equ 0 (
                        set "PYTHON_CMD=!PYTHON_EXE!"
                        set "PYTHON_FOUND=1"
                        echo [SUCCESS] Found working Python: !PYTHON_EXE!
                        goto :run_program
                    )
                )
            )
        )
    )
)

REM Method 5: Use where command to find all Python installations
echo Searching all Python installations...
for /f "tokens=*" %%p in ('where python.exe 2^>nul') do (
    echo Testing: %%p
    "%%p" -c "import markdown; import pymilvus" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=%%p"
        set "PYTHON_FOUND=1"
        echo [SUCCESS] Found working Python: %%p
        goto :run_program
    )
)

REM If no working Python found
if !PYTHON_FOUND! equ 0 (
    echo.
    echo ================================================================
    echo ERROR: No Python environment found with required packages
    echo ================================================================
    echo.
    echo This usually happens when running as administrator and:
    echo 1. Packages are installed only for the current user
    echo 2. Different Python environment is being used
    echo.
    echo SOLUTIONS:
    echo.
    echo Option 1: Install packages for all users (as administrator)
    echo   python -m pip install --upgrade markdown pymilvus sentence-transformers
    echo.
    echo Option 2: Run without administrator privileges
    echo   Right-click run-main.bat and select "Run as administrator" is not needed
    echo   Most MCP operations work with normal user privileges
    echo.
    echo Option 3: Use run-main.py instead
    echo   python run-main.py
    echo.
    echo Run diagnose-python-env.bat for detailed environment information
    echo.
    pause
    exit /b 1
)

:run_program
echo.
echo ================================================================
echo Starting program with: %PYTHON_CMD%
echo ================================================================
echo.

REM Show Python environment info
echo Python environment:
"%PYTHON_CMD%" --version
echo Executable: 
"%PYTHON_CMD%" -c "import sys; print(sys.executable)"
echo.

echo Starting main.py...
echo.

REM Run main.py with the found Python
"%PYTHON_CMD%" main.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: main.py exited with error code %errorlevel%
    echo.
)

echo.
echo Program finished.
pause