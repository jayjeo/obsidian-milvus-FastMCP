@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo       Podman Auto-Startup Setup (Administrator Required)
echo ================================================================
echo.

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

REM Check common Podman installation paths
set "PODMAN_PATHS=C:\Program Files\RedHat\Podman\podman.exe C:\Program Files (x86)\RedHat\Podman\podman.exe"
for %%p in (%PODMAN_PATHS%) do (
    if exist "%%p" (
        set "PODMAN_PATH=%%p"
        echo Found Podman: !PODMAN_PATH!
        goto :podman_found
    )
)

REM Try to find Podman in PATH
for /f "tokens=*" %%p in ('where podman 2^>nul') do (
    set "PODMAN_PATH=%%p"
    echo Found Podman in PATH: !PODMAN_PATH!
    goto :podman_found
)

echo ERROR: Podman not found. Please install Podman first.
echo Download from: https://podman.io/
pause
exit /b 1

:podman_found
echo Testing Podman...
"!PODMAN_PATH!" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Podman found but not working properly.
    pause
    exit /b 1
)

echo Podman is working: 
"!PODMAN_PATH!" --version

REM Create VBS script for auto-startup
echo.
echo Creating VBS startup script...

set "VBS_SCRIPT_PATH=%SCRIPT_DIR%\podman_auto_startup.vbs"

echo ' ================================================================ > "%VBS_SCRIPT_PATH%"
echo ' Podman Auto-Startup Script for Windows >> "%VBS_SCRIPT_PATH%"
echo ' This script starts Podman machine and Milvus containers >> "%VBS_SCRIPT_PATH%"
echo ' ================================================================ >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo Dim fso, shell, projectDir >> "%VBS_SCRIPT_PATH%"
echo Set fso = CreateObject("Scripting.FileSystemObject") >> "%VBS_SCRIPT_PATH%"
echo Set shell = CreateObject("WScript.Shell") >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Get project directory from script location >> "%VBS_SCRIPT_PATH%"
echo projectDir = fso.GetParentFolderName(WScript.ScriptFullName) >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Define paths >> "%VBS_SCRIPT_PATH%"
echo Dim podmanPath, logFile, startupComplete, composeFile >> "%VBS_SCRIPT_PATH%"
echo podmanPath = "%PODMAN_PATH:\=\\%" >> "%VBS_SCRIPT_PATH%"
echo logFile = projectDir ^& "\\podman_startup.log" >> "%VBS_SCRIPT_PATH%"
echo startupComplete = projectDir ^& "\\startup_complete.flag" >> "%VBS_SCRIPT_PATH%"
echo composeFile = projectDir ^& "\\milvus-podman-compose.yml" >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Function to write log entries >> "%VBS_SCRIPT_PATH%"
echo Sub WriteLog(message) >> "%VBS_SCRIPT_PATH%"
echo     Dim logFileHandle >> "%VBS_SCRIPT_PATH%"
echo     Set logFileHandle = fso.OpenTextFile(logFile, 8, True) >> "%VBS_SCRIPT_PATH%"
echo     logFileHandle.WriteLine Now ^& " - " ^& message >> "%VBS_SCRIPT_PATH%"
echo     logFileHandle.Close >> "%VBS_SCRIPT_PATH%"
echo End Sub >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Function to run command silently >> "%VBS_SCRIPT_PATH%"
echo Function RunCommand(cmd) >> "%VBS_SCRIPT_PATH%"
echo     RunCommand = shell.Run(cmd, 0, True) >> "%VBS_SCRIPT_PATH%"
echo End Function >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Main startup process >> "%VBS_SCRIPT_PATH%"
echo Sub Main() >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "==========================================" >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Podman Auto-Startup Script Started" >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Project Directory: " ^& projectDir >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Podman Path: " ^& podmanPath >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "==========================================" >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Wait for system to be ready >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Waiting 30 seconds for system initialization..." >> "%VBS_SCRIPT_PATH%"
echo     WScript.Sleep 30000 >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Check if Podman executable exists >> "%VBS_SCRIPT_PATH%"
echo     If Not fso.FileExists(podmanPath) Then >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "ERROR: Podman not found at " ^& podmanPath >> "%VBS_SCRIPT_PATH%"
echo         Exit Sub >> "%VBS_SCRIPT_PATH%"
echo     End If >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Start Podman machine (Windows may need this) >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Starting Podman machine..." >> "%VBS_SCRIPT_PATH%"
echo     Dim result >> "%VBS_SCRIPT_PATH%"
echo     result = RunCommand("""" ^& podmanPath ^& """ machine start") >> "%VBS_SCRIPT_PATH%"
echo     If result = 0 Then >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "Podman machine started successfully" >> "%VBS_SCRIPT_PATH%"
echo     Else >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "Podman machine start returned code: " ^& result ^& " (may already be running)" >> "%VBS_SCRIPT_PATH%"
echo     End If >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Wait for Podman machine to be ready >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Waiting 20 seconds for Podman machine to be ready..." >> "%VBS_SCRIPT_PATH%"
echo     WScript.Sleep 20000 >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Check if compose file exists and start containers >> "%VBS_SCRIPT_PATH%"
echo     If fso.FileExists(composeFile) Then >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "Starting Milvus containers using compose file..." >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo         ' Change to project directory and start containers >> "%VBS_SCRIPT_PATH%"
echo         shell.CurrentDirectory = projectDir >> "%VBS_SCRIPT_PATH%"
echo         result = RunCommand("""" ^& podmanPath ^& """ compose -f """ ^& composeFile ^& """ up -d") >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo         If result = 0 Then >> "%VBS_SCRIPT_PATH%"
echo             WriteLog "Milvus containers started successfully" >> "%VBS_SCRIPT_PATH%"
echo         Else >> "%VBS_SCRIPT_PATH%"
echo             WriteLog "Container startup returned code: " ^& result >> "%VBS_SCRIPT_PATH%"
echo         End If >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo         ' Additional wait for containers to be ready >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "Waiting 30 seconds for containers to be ready..." >> "%VBS_SCRIPT_PATH%"
echo         WScript.Sleep 30000 >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     Else >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "WARNING: Compose file not found: " ^& composeFile >> "%VBS_SCRIPT_PATH%"
echo         WriteLog "Skipping container startup" >> "%VBS_SCRIPT_PATH%"
echo     End If >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     ' Create completion flag >> "%VBS_SCRIPT_PATH%"
echo     Dim flagFile >> "%VBS_SCRIPT_PATH%"
echo     Set flagFile = fso.CreateTextFile(startupComplete, True) >> "%VBS_SCRIPT_PATH%"
echo     flagFile.WriteLine "Startup completed at " ^& Now >> "%VBS_SCRIPT_PATH%"
echo     flagFile.Close >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo     WriteLog "Startup process completed" >> "%VBS_SCRIPT_PATH%"
echo End Sub >> "%VBS_SCRIPT_PATH%"
echo. >> "%VBS_SCRIPT_PATH%"
echo ' Run the main process >> "%VBS_SCRIPT_PATH%"
echo Main >> "%VBS_SCRIPT_PATH%"

echo VBS script created: %VBS_SCRIPT_PATH%
echo.

REM Register with Windows Task Scheduler
echo Registering with Windows Task Scheduler...

REM Create the task
schtasks /create /tn "PodmanAutoStartup" /tr "wscript.exe \"%VBS_SCRIPT_PATH%\"" /sc onstart /ru SYSTEM /rl highest /f /st 00:00 /sd 01/01/2024

if %errorlevel% neq 0 (
    echo Failed to create scheduled task. Error code: %errorlevel%
    echo This might be due to insufficient privileges.
    echo Please make sure you're running this script as Administrator.
) else (
    echo Task created successfully!
    echo Podman will now start automatically when Windows boots.
    echo.
    echo You can verify this in Task Scheduler:
    echo 1. Open Task Scheduler (taskschd.msc)
    echo 2. Look for the "PodmanAutoStartup" task
)

echo.
echo Setup completed.
pause
