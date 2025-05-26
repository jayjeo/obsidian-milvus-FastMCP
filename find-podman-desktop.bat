@echo off
chcp 65001 >nul
echo ================================================================
echo           Finding and Starting Podman Desktop
echo ================================================================
echo.

echo [INFO] Searching for Podman Desktop installation...

REM Check common installation paths for Podman Desktop
set PODMAN_DESKTOP_PATH=
set "DESKTOP_PATHS_TO_CHECK="C:\Program Files\Podman Desktop" "C:\Program Files (x86)\Podman Desktop" "%LOCALAPPDATA%\Programs\Podman Desktop" "%APPDATA%\Local\Programs\Podman Desktop" "C:\Users\%USERNAME%\AppData\Local\Programs\Podman Desktop""

for %%P in (%DESKTOP_PATHS_TO_CHECK%) do (
    if exist "%%~P\Podman Desktop.exe" (
        set "PODMAN_DESKTOP_PATH=%%~P\Podman Desktop.exe"
        echo [FOUND] Podman Desktop at: %%~P
        goto :found_desktop
    )
    if exist "%%~P\podman-desktop.exe" (
        set "PODMAN_DESKTOP_PATH=%%~P\podman-desktop.exe"
        echo [FOUND] Podman Desktop at: %%~P
        goto :found_desktop
    )
)

REM Search in more locations
echo [INFO] Searching in additional locations...
for /f "tokens=*" %%i in ('dir /s /b "C:\Program Files\*podman*desktop*.exe" 2^>nul') do (
    echo [FOUND] Podman Desktop candidate: %%i
    set "PODMAN_DESKTOP_PATH=%%i"
    goto :found_desktop
)

for /f "tokens=*" %%i in ('dir /s /b "C:\Users\%USERNAME%\AppData\Local\Programs\*podman*desktop*.exe" 2^>nul') do (
    echo [FOUND] Podman Desktop candidate: %%i
    set "PODMAN_DESKTOP_PATH=%%i"
    goto :found_desktop
)

REM Check Windows Registry for installed programs
echo [INFO] Checking Windows Registry for Podman Desktop...
for /f "tokens=2*" %%A in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" /s /f "Podman Desktop" 2^>nul ^| findstr "DisplayName"') do (
    echo [FOUND] Registry entry: %%B
)

for /f "tokens=2*" %%A in ('reg query "HKEY_CURRENT_USER\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" /s /f "Podman Desktop" 2^>nul ^| findstr "DisplayName"') do (
    echo [FOUND] Registry entry: %%B
)

if not defined PODMAN_DESKTOP_PATH (
    echo [WARNING] Podman Desktop executable not found in common locations
    echo.
    echo Possible solutions:
    echo 1. Download and install Podman Desktop from: https://podman-desktop.io/
    echo 2. Or manually install WSL2 and use command line setup
    echo 3. Check if it's installed with a different name
    echo.
    goto :manual_wsl_setup
)

:found_desktop
echo.
echo [INFO] Starting Podman Desktop...
echo Path: %PODMAN_DESKTOP_PATH%
echo.

start "" "%PODMAN_DESKTOP_PATH%"
if %errorlevel% equ 0 (
    echo [SUCCESS] Podman Desktop should be starting...
    echo.
    echo [NEXT STEPS]
    echo 1. Wait for Podman Desktop to fully load
    echo 2. It may prompt you to install/configure WSL2
    echo 3. Follow the setup wizard in the GUI
    echo 4. Once setup is complete, come back and run run-setup.bat
    echo.
    goto :end
) else (
    echo [ERROR] Failed to start Podman Desktop
    goto :manual_wsl_setup
)

:manual_wsl_setup
echo.
echo ================================================================
echo              Alternative: Manual WSL2 Setup
echo ================================================================
echo.
echo Since Podman Desktop is not readily available, let's try manual setup:
echo.

choice /c YN /m "Do you want to install WSL2 manually"
if errorlevel 2 goto :end

echo.
echo [INFO] Installing WSL2 (requires administrator privileges)...
echo [WARNING] You may need to restart your computer after this
echo.

powershell -Command "Start-Process cmd -ArgumentList '/c wsl --install' -Verb RunAs"
if %errorlevel% equ 0 (
    echo [INFO] WSL2 installation initiated
    echo.
    echo [IMPORTANT] After installation completes:
    echo 1. Restart your computer if prompted
    echo 2. Run this script again to set up Podman machine
    echo 3. Or run: podman machine init && podman machine start
) else (
    echo [ERROR] Failed to start WSL2 installation
    echo.
    echo [MANUAL] Please try:
    echo 1. Open PowerShell as Administrator
    echo 2. Run: wsl --install
    echo 3. Restart computer if prompted
)

:end
echo.
echo ================================================================
echo                         Summary
echo ================================================================
echo.
echo Current status:
echo - Podman CLI: Installed and working
echo - WSL2: Needs installation
echo - Podman Machine: Not created yet
echo.
echo Next steps:
echo 1. Install WSL2 (via Podman Desktop or manually)
echo 2. Create Podman machine
echo 3. Run run-setup.bat option 2
echo.
pause
