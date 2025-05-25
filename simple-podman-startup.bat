@echo off
setlocal

echo ================================================================
echo       Simple Podman Auto-Startup Setup
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script requires administrator privileges.
    echo Please right-click on this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Find Podman executable
echo Finding Podman installation...
set "PODMAN_PATH="

if exist "C:\Program Files\RedHat\Podman\podman.exe" (
    set "PODMAN_PATH=C:\Program Files\RedHat\Podman\podman.exe"
    echo Found Podman: !PODMAN_PATH!
    goto :found_podman
)

if exist "C:\Program Files (x86)\RedHat\Podman\podman.exe" (
    set "PODMAN_PATH=C:\Program Files (x86)\RedHat\Podman\podman.exe"
    echo Found Podman: !PODMAN_PATH!
    goto :found_podman
)

for /f "tokens=*" %%p in ('where podman 2^>nul') do (
    set "PODMAN_PATH=%%p"
    echo Found Podman in PATH: !PODMAN_PATH!
    goto :found_podman
)

echo ERROR: Podman not found. Please install Podman first.
echo Download from: https://podman.io/
pause
exit /b 1

:found_podman
echo Testing Podman...
"%PODMAN_PATH%" --version
if %errorlevel% neq 0 (
    echo ERROR: Podman found but not working properly.
    pause
    exit /b 1
)

REM Create simple VBS script
echo.
echo Creating VBS startup script...

set "VBS_PATH=%SCRIPT_DIR%\simple_podman_startup.vbs"

(
echo ' Podman Auto-Startup Script
echo Set shell = CreateObject^("WScript.Shell"^)
echo Set fso = CreateObject^("Scripting.FileSystemObject"^)
echo projectDir = fso.GetParentFolderName^(WScript.ScriptFullName^)
echo podmanPath = "%PODMAN_PATH:\=\\%"
echo logFile = projectDir ^& "\\podman_startup.log"
echo.
echo ' Write to log
echo Set logFileHandle = fso.OpenTextFile^(logFile, 8, True^)
echo logFileHandle.WriteLine Now ^& " - Starting Podman"
echo logFileHandle.Close
echo.
echo ' Wait for system to be ready
echo WScript.Sleep 30000
echo.
echo ' Start Podman machine
echo shell.Run """" ^& podmanPath ^& """ machine start", 0, True
echo.
echo ' Wait for machine to be ready
echo WScript.Sleep 20000
echo.
echo ' Start containers
echo shell.CurrentDirectory = projectDir
echo shell.Run """" ^& podmanPath ^& """ compose -f """ ^& projectDir ^& "\\milvus-podman-compose.yml"" up -d", 0, True
echo.
echo ' Write completion to log
echo Set logFileHandle = fso.OpenTextFile^(logFile, 8, True^)
echo logFileHandle.WriteLine Now ^& " - Startup completed"
echo logFileHandle.Close
) > "%VBS_PATH%"

echo VBS script created: %VBS_PATH%
echo.

REM Register with Windows Task Scheduler
echo Registering with Windows Task Scheduler...

schtasks /create /tn "PodmanAutoStartup" /tr "wscript.exe \"%VBS_PATH%\"" /sc onstart /ru SYSTEM /rl highest /f

if %errorlevel% neq 0 (
    echo Failed to create scheduled task. Error code: %errorlevel%
    echo This might be due to insufficient privileges.
    echo Please make sure you're running this script as Administrator.
) else (
    echo Task created successfully!
    echo Podman will now start automatically when Windows boots.
)

echo.
echo Setup completed.
pause
