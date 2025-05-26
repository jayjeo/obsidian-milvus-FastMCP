@echo off
chcp 65001 >nul
echo ================================================================
echo           WSL2 Installation and Podman Setup Helper
echo ================================================================
echo.

echo [INFO] Checking WSL installation status...
wsl --status >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] WSL is installed
    wsl --list --verbose
    goto :check_podman_machine
) else (
    echo [WARNING] WSL is not installed or not properly configured
)

echo.
echo [INFO] Checking if WSL feature is enabled...
dism /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] WSL feature is available
) else (
    echo [WARNING] WSL feature may not be available
)

echo.
echo ================================================================
echo                    WSL2 Installation Options
echo ================================================================
echo.
echo Option 1 (Recommended): Use Podman Desktop
echo   - Run "Podman Desktop" from Start Menu
echo   - It will automatically handle WSL2 installation
echo.
echo Option 2: Manual WSL2 Installation
echo   - Requires administrator privileges
echo   - Will install WSL2 and Ubuntu distribution
echo.

choice /c 12Q /m "Choose option [1=Podman Desktop, 2=Manual Install, Q=Quit]"

if errorlevel 3 goto :end
if errorlevel 2 goto :manual_wsl_install
if errorlevel 1 goto :open_podman_desktop

:open_podman_desktop
echo.
echo [INFO] Attempting to open Podman Desktop...
start "" "Podman Desktop" 2>nul
if %errorlevel% equ 0 (
    echo [OK] Podman Desktop should be opening
    echo [NEXT] Use the GUI to set up WSL2 and Podman machine
) else (
    echo [WARNING] Could not open Podman Desktop automatically
    echo [MANUAL] Please open "Podman Desktop" from Start Menu manually
)
goto :end

:manual_wsl_install
echo.
echo [WARNING] Manual WSL2 installation requires administrator privileges
echo [INFO] You may need to run this as administrator
echo.

choice /c YN /m "Continue with manual WSL2 installation"
if errorlevel 2 goto :end

echo.
echo [INFO] Installing WSL2... This may take several minutes
echo [INFO] You may be prompted for administrator permissions
echo.

wsl --install
if %errorlevel% equ 0 (
    echo [OK] WSL installation command executed
    echo [IMPORTANT] You may need to restart your computer
    echo [NEXT] After restart, run this script again
) else (
    echo [ERROR] WSL installation failed
    echo [SOLUTION] Try running as administrator or use Podman Desktop
)

goto :end

:check_podman_machine
echo.
echo [INFO] WSL is available. Trying to create Podman machine...
echo.

podman machine init
if %errorlevel% equ 0 (
    echo [OK] Podman machine created successfully
    echo.
    echo [INFO] Starting Podman machine...
    podman machine start
    if %errorlevel% equ 0 (
        echo [SUCCESS] Podman machine is running!
        echo [NEXT] You can now run run-setup.bat and select option 2
    ) else (
        echo [WARNING] Machine created but failed to start
        echo [SOLUTION] Try: podman machine start
    )
) else (
    echo [ERROR] Podman machine creation still failed
    echo [SOLUTION] Use Podman Desktop for easier setup
)

:end
echo.
echo ================================================================
echo                         Summary
echo ================================================================
echo.
echo For the easiest setup experience:
echo 1. Open "Podman Desktop" from Start Menu
echo 2. Follow the GUI setup wizard
echo 3. It will handle WSL2 and machine setup automatically
echo.
echo Alternative: Install WSL2 manually then retry Podman setup
echo.
pause
