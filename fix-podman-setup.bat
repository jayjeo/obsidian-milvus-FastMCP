@echo off
chcp 65001 >nul
echo ================================================================
echo            Podman Setup Fix Tool for Windows
echo ================================================================
echo.

REM Check if podman is already in PATH
where podman >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Podman is already in PATH
    podman --version
    goto :check_machine
)

echo [WARNING] Podman not found in PATH. Searching installation paths...
echo.

REM Check common Podman installation paths
set PODMAN_PATH=
set "PATHS_TO_CHECK="C:\Program Files\RedHat\Podman" "C:\Program Files (x86)\RedHat\Podman" "C:\Program Files\Podman" "C:\Program Files (x86)\Podman" "%LOCALAPPDATA%\RedHat\Podman" "%APPDATA%\RedHat\Podman""

for %%P in (%PATHS_TO_CHECK%) do (
    if exist "%%~P\podman.exe" (
        set "PODMAN_PATH=%%~P"
        echo [FOUND] Podman at: %%~P
        "%%~P\podman.exe" --version
        goto :add_to_path
    )
)

if not defined PODMAN_PATH (
    echo [ERROR] Podman executable not found in common locations
    echo.
    echo Solutions:
    echo 1. Check if Podman Desktop is installed
    echo 2. Run Podman Desktop from Start Menu
    echo 3. Or reinstall from https://podman.io/getting-started/installation
    echo.
    goto :end
)

:add_to_path
echo.
echo [INFO] Adding Podman to PATH for current session...
echo Path: %PODMAN_PATH%
echo.

REM Add to PATH for current session
set "PATH=%PATH%;%PODMAN_PATH%"
echo [OK] PATH updated for current session

REM Test if podman works now
podman --version >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Podman command is now available
) else (
    echo [ERROR] Podman command still fails
    goto :end
)

:check_machine
echo.
echo ================================================================
echo               Checking Podman Machine Status
echo ================================================================

echo [INFO] Checking existing machines...
podman machine list

REM Check if any machine exists and try to start
for /f "skip=1 tokens=1" %%i in ('podman machine list 2^>nul') do (
    if not "%%i"=="" (
        echo [FOUND] Existing machine: %%i
        echo.
        echo [INFO] Attempting to start machine...
        podman machine start %%i
        if %errorlevel% equ 0 (
            echo [OK] Machine started successfully
            goto :test_connection
        ) else (
            echo [WARNING] Machine start failed - continuing anyway
        )
        goto :test_connection
    )
)

echo [WARNING] No Podman machine found. Creating new one...
echo.
echo NOTE: This operation requires:
echo    - WSL2 to be installed
echo    - Virtualization to be enabled
echo    - May take several minutes
echo.

choice /c YN /m "Do you want to create a Podman machine"
if errorlevel 2 goto :manual_setup

echo [INFO] Creating default Podman machine...
podman machine init
if %errorlevel% equ 0 (
    echo [OK] Machine created successfully
    echo.
    echo [INFO] Starting machine...
    podman machine start
    if %errorlevel% equ 0 (
        echo [OK] Machine started successfully
    ) else (
        echo [ERROR] Machine start failed
        goto :troubleshoot
    )
) else (
    echo [ERROR] Machine creation failed
    goto :troubleshoot
)

:test_connection
echo.
echo ================================================================
echo               Testing Podman Connection
echo ================================================================

echo [TEST] Basic connection test...
podman info >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Podman service connection successful
) else (
    echo [ERROR] Podman service connection failed
    goto :troubleshoot
)

echo.
echo [TEST] Network functionality test...
podman network ls >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Network functionality normal
) else (
    echo [ERROR] Network functionality has issues
    goto :troubleshoot
)

echo.
echo ================================================================
echo                    Setup Complete!
echo ================================================================
echo.
echo [SUCCESS] Podman has been configured successfully!
echo [NEXT] Now run run-setup.bat and select option 2
echo.
goto :end

:troubleshoot
echo.
echo ================================================================
echo                   Troubleshooting Guide
echo ================================================================
echo.
echo [ERROR] There are issues with Podman setup.
echo.
echo Solutions:
echo.
echo 1. Use Podman Desktop (Recommended):
echo    - Run "Podman Desktop" from Start Menu
echo    - Configure and start machine via GUI
echo.
echo 2. Check WSL2:
echo    - Run 'wsl --list --verbose' in PowerShell
echo    - If WSL2 not installed: 'wsl --install'
echo.
echo 3. Check Virtualization:
echo    - Enable virtualization (VT-x/AMD-V) in BIOS
echo    - Enable 'Hyper-V' or 'WSL' in Windows Features
echo.
echo 4. Manual Setup:
echo    - Run 'podman machine init --now'
echo    - Run 'podman machine start'
echo.
goto :end

:manual_setup
echo.
echo Manual Setup Instructions:
echo.
echo 1. Run Podman Desktop application
echo 2. Or use commands: podman machine init
echo 3. Then: podman machine start
echo.

:end
echo.
echo ================================================================
echo Done! Use Podman Desktop GUI for additional help if needed.
echo ================================================================
pause
