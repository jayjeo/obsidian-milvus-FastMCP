@echo off
setlocal enabledelayedexpansion
title Podman Compose Fix Tool

echo ================================================================
echo                    Podman Compose Fix Tool
echo ================================================================
echo This script will diagnose and fix Podman compose issues
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Step 1: Checking Podman installation...
echo --------------------------------

:: Check if Podman is installed
"C:\Program Files\RedHat\Podman\podman.exe" --version
if errorlevel 1 (
    echo ERROR: Podman not found at standard location
    echo Please install Podman Desktop first
    pause
    exit /b 1
)

echo Podman found successfully
echo.

echo Step 2: Checking Podman machine status...
echo -----------------------------------------

"C:\Program Files\RedHat\Podman\podman.exe" machine list
echo.

echo Step 3: Testing Podman compose support...
echo -----------------------------------------

:: Test if podman compose works
echo Testing: podman compose --help
"C:\Program Files\RedHat\Podman\podman.exe" compose --help >nul 2>&1
if errorlevel 1 (
    echo ISSUE: Built-in podman compose not working
    echo This is the cause of your error
    goto install_podman_compose
) else (
    echo SUCCESS: Built-in podman compose is available
    goto test_compose_file
)

:install_podman_compose
echo.
echo Step 4: Installing podman-compose as fallback...
echo ------------------------------------------------

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found in PATH
        echo Please install Python first
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

echo Python found: %PYTHON_CMD%
echo Installing podman-compose...

%PYTHON_CMD% -m pip install podman-compose
if errorlevel 1 (
    echo ERROR: Failed to install podman-compose
    echo Please run this as Administrator or check internet connection
    pause
    exit /b 1
)

echo podman-compose installed successfully
echo.

:: Test podman-compose
echo Testing podman-compose...
podman-compose --version
if errorlevel 1 (
    echo WARNING: podman-compose not found in PATH after installation
    echo You may need to restart your command prompt or add Python Scripts to PATH
)

goto test_compose_file

:test_compose_file
echo.
echo Step 5: Testing compose with Milvus configuration...
echo ---------------------------------------------------

if exist "milvus-podman-compose.yml" (
    echo Testing compose file validation...
    
    :: Try built-in compose first
    "C:\Program Files\RedHat\Podman\podman.exe" compose -f milvus-podman-compose.yml config >nul 2>&1
    if errorlevel 1 (
        echo Built-in compose validation failed, trying podman-compose...
        podman-compose -f milvus-podman-compose.yml config >nul 2>&1
        if errorlevel 1 (
            echo ERROR: Both compose methods failed to validate the configuration
            echo There may be an issue with the compose file
        ) else (
            echo SUCCESS: podman-compose can read the configuration
        )
    ) else (
        echo SUCCESS: Built-in compose can read the configuration
    )
) else (
    echo WARNING: milvus-podman-compose.yml not found in current directory
    echo Make sure you're running this from the project directory
)

echo.
echo Step 6: Testing Milvus startup (DRY RUN)...
echo -------------------------------------------

echo This will test starting Milvus containers without actually starting them
echo.

:: Try to bring containers up in detached mode (dry run with config check)
echo Testing startup command...
"C:\Program Files\RedHat\Podman\podman.exe" compose -f milvus-podman-compose.yml up --dry-run 2>nul
if errorlevel 1 (
    echo Built-in compose dry run failed, trying podman-compose...
    podman-compose -f milvus-podman-compose.yml up --help >nul 2>&1
    if errorlevel 1 (
        echo ERROR: podman-compose is not working properly
    ) else (
        echo SUCCESS: podman-compose should work for starting containers
    )
) else (
    echo SUCCESS: Built-in compose should work for starting containers
)

echo.
echo ================================================================
echo                          SUMMARY
echo ================================================================

if exist "%USERPROFILE%\.local\bin\podman-compose.exe" (
    echo STATUS: podman-compose appears to be installed
) else (
    echo STATUS: Checking if podman-compose is in PATH...
    podman-compose --version >nul 2>&1
    if errorlevel 1 (
        echo STATUS: podman-compose installation may have failed
        echo RECOMMENDATION: Try running 'pip install --user podman-compose'
    ) else (
        echo STATUS: podman-compose is working
    )
)

echo.
echo Next steps:
echo 1. If podman-compose was installed, restart your command prompt
echo 2. Try running main.py again
echo 3. If issues persist, try running start-milvus.bat manually
echo.
echo Files to check:
echo - vbs_startup.log (VBS script logs)
echo - auto_startup_mcp.log (Python script logs)
echo - podman_startup.log (Podman startup logs)
echo.

pause
