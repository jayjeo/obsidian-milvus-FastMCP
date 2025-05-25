@echo off
echo ================================================================
echo                    Quick Podman Finder
echo ================================================================
echo.

REM Quick check - is podman already in PATH?
where podman.exe >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Podman is already configured!
    echo.
    echo Location:
    where podman.exe
    echo.
    echo Version:
    podman --version
    echo.
    echo No further configuration needed.
    goto :end
)

echo Podman not in PATH. Searching common locations...
echo.

REM Check most common locations
set "FOUND=0"

if exist "%ProgramFiles%\RedHat\Podman\podman.exe" (
    echo [FOUND] %ProgramFiles%\RedHat\Podman\podman.exe
    set "FOUND=1"
    set "PODMAN_PATH=%ProgramFiles%\RedHat\Podman"
)

if exist "%ProgramFiles(x86)%\RedHat\Podman\podman.exe" (
    echo [FOUND] %ProgramFiles(x86)%\RedHat\Podman\podman.exe
    set "FOUND=1"
    set "PODMAN_PATH=%ProgramFiles(x86)%\RedHat\Podman"
)

if exist "%LOCALAPPDATA%\Podman\podman.exe" (
    echo [FOUND] %LOCALAPPDATA%\Podman\podman.exe
    set "FOUND=1"
    set "PODMAN_PATH=%LOCALAPPDATA%\Podman"
)

if %FOUND% == 1 (
    echo.
    echo [SUCCESS] Podman found but not in PATH.
    echo.
    echo To fix this, run one of these commands as Administrator:
    echo.
    echo For current user only:
    echo   setx PATH "%%PATH%%;%PODMAN_PATH%"
    echo.
    echo Or manually add this directory to your PATH:
    echo   %PODMAN_PATH%
    echo.
    echo After adding to PATH, restart your command prompt and type:
    echo   podman --version
) else (
    echo.
    echo [NOT FOUND] Podman not found in common locations.
    echo.
    echo Run find_podman_advanced.bat for comprehensive search.
    echo.
    echo Or install Podman from: https://podman.io/
)

:end
echo.
echo ================================================================
pause
