@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo          Podman Auto-Startup Setup for Windows
echo ================================================================
echo.
echo This script will:
echo 1. Find Podman executable automatically
echo 2. Create VBS script for silent startup
echo 3. Register with Windows Task Scheduler
echo 4. Setup automatic Podman startup on Windows boot
echo.

REM Get current project directory
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

echo Project directory: %PROJECT_DIR%
echo.

REM Step 1: Find Podman executable
echo ================================================================
echo Step 1: Finding Podman executable
echo ================================================================

set "PODMAN_PATH="

REM Check if Podman is in PATH first
where podman.exe >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('where podman.exe 2^>nul') do (
        set "PODMAN_PATH=%%i"
        echo [SUCCESS] Found Podman in PATH: !PODMAN_PATH!
        goto :found_podman
    )
)

REM Check common installation paths
echo Searching common installation directories...
set "SEARCH_PATHS[0]=%ProgramFiles%\RedHat\Podman\podman.exe"
set "SEARCH_PATHS[1]=%ProgramFiles(x86)%\RedHat\Podman\podman.exe"
set "SEARCH_PATHS[2]=%LOCALAPPDATA%\Podman\podman.exe"
set "SEARCH_PATHS[3]=%ProgramFiles%\Podman\podman.exe"
set "SEARCH_PATHS[4]=%ProgramFiles(x86)%\Podman\podman.exe"

for /L %%i in (0,1,4) do (
    if defined SEARCH_PATHS[%%i] (
        if exist "!SEARCH_PATHS[%%i]!" (
            set "PODMAN_PATH=!SEARCH_PATHS[%%i]!"
            echo [SUCCESS] Found Podman at: !PODMAN_PATH!
            goto :found_podman
        )
    )
)

echo [ERROR] Podman not found!
echo.
echo Please install Podman first:
echo 1. Download from: https://podman.io/
echo 2. Or install Podman Desktop
echo.
echo After installation, run this script again.
pause
exit /b 1

:found_podman
echo.
echo Testing Podman functionality...
"!PODMAN_PATH!" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Podman found but not working properly
    echo Please check your Podman installation
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('"!PODMAN_PATH!" --version 2^>nul') do (
    echo [SUCCESS] Podman is working: %%v
)

REM Step 2: Create VBS startup script
echo.
echo ================================================================
echo Step 2: Creating VBS startup script
echo ================================================================

set "VBS_SCRIPT=%PROJECT_DIR%\podman_auto_startup.vbs"

echo Creating: %VBS_SCRIPT%

REM Create VBS script content
echo ' ================================================================> "%VBS_SCRIPT%"
echo ' Podman Auto-Startup Script for Windows>> "%VBS_SCRIPT%"
echo ' This script starts Podman machine and Milvus containers>> "%VBS_SCRIPT%"
echo ' Runs silently at Windows startup>> "%VBS_SCRIPT%"
echo ' ================================================================>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo Dim fso, shell, projectDir>> "%VBS_SCRIPT%"
echo Set fso = CreateObject("Scripting.FileSystemObject")>> "%VBS_SCRIPT%"
echo Set shell = CreateObject("WScript.Shell")>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Get project directory from script location>> "%VBS_SCRIPT%"
echo projectDir = fso.GetParentFolderName(WScript.ScriptFullName)>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Define paths (using relative paths from config.py)>> "%VBS_SCRIPT%"
echo Dim podmanPath, logFile, startupComplete>> "%VBS_SCRIPT%"
echo podmanPath = "!PODMAN_PATH!">> "%VBS_SCRIPT%"
echo logFile = projectDir ^& "\podman_startup.log">> "%VBS_SCRIPT%"
echo startupComplete = projectDir ^& "\startup_complete.flag">> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Function to write log entries>> "%VBS_SCRIPT%"
echo Sub WriteLog(message)>> "%VBS_SCRIPT%"
echo     Dim logFileHandle>> "%VBS_SCRIPT%"
echo     Set logFileHandle = fso.OpenTextFile(logFile, 8, True)>> "%VBS_SCRIPT%"
echo     logFileHandle.WriteLine Now ^& " - " ^& message>> "%VBS_SCRIPT%"
echo     logFileHandle.Close>> "%VBS_SCRIPT%"
echo End Sub>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Function to run command silently>> "%VBS_SCRIPT%"
echo Function RunCommand(cmd)>> "%VBS_SCRIPT%"
echo     RunCommand = shell.Run(cmd, 0, True)>> "%VBS_SCRIPT%"
echo End Function>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Main startup process>> "%VBS_SCRIPT%"
echo Sub Main()>> "%VBS_SCRIPT%"
echo     WriteLog "==========================================">> "%VBS_SCRIPT%"
echo     WriteLog "Podman Auto-Startup Script Started">> "%VBS_SCRIPT%"
echo     WriteLog "Project Directory: " ^& projectDir>> "%VBS_SCRIPT%"
echo     WriteLog "Podman Path: " ^& podmanPath>> "%VBS_SCRIPT%"
echo     WriteLog "==========================================">> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Wait for system to be ready>> "%VBS_SCRIPT%"
echo     WriteLog "Waiting 30 seconds for system initialization...">> "%VBS_SCRIPT%"
echo     WScript.Sleep 30000>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Check if Podman executable exists>> "%VBS_SCRIPT%"
echo     If Not fso.FileExists(podmanPath) Then>> "%VBS_SCRIPT%"
echo         WriteLog "ERROR: Podman not found at " ^& podmanPath>> "%VBS_SCRIPT%"
echo         Exit Sub>> "%VBS_SCRIPT%"
echo     End If>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Start Podman machine (Windows/macOS may need this)>> "%VBS_SCRIPT%"
echo     WriteLog "Starting Podman machine...">> "%VBS_SCRIPT%"
echo     Dim result>> "%VBS_SCRIPT%"
echo     result = RunCommand("""" ^& podmanPath ^& """ machine start")>> "%VBS_SCRIPT%"
echo     If result = 0 Then>> "%VBS_SCRIPT%"
echo         WriteLog "Podman machine started successfully">> "%VBS_SCRIPT%"
echo     Else>> "%VBS_SCRIPT%"
echo         WriteLog "Podman machine start returned code: " ^& result ^& " (may already be running)">> "%VBS_SCRIPT%"
echo     End If>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Wait for Podman machine to be ready>> "%VBS_SCRIPT%"
echo     WriteLog "Waiting 15 seconds for Podman machine to be ready...">> "%VBS_SCRIPT%"
echo     WScript.Sleep 15000>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Check if compose file exists and start containers>> "%VBS_SCRIPT%"
echo     Dim composeFile>> "%VBS_SCRIPT%"
echo     composeFile = projectDir ^& "\milvus-podman-compose.yml">> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     If fso.FileExists(composeFile) Then>> "%VBS_SCRIPT%"
echo         WriteLog "Starting Milvus containers using compose file...">> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo         ' Change to project directory and start containers>> "%VBS_SCRIPT%"
echo         shell.CurrentDirectory = projectDir>> "%VBS_SCRIPT%"
echo         result = RunCommand("""" ^& podmanPath ^& """ compose -f """ ^& composeFile ^& """ up -d")>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo         If result = 0 Then>> "%VBS_SCRIPT%"
echo             WriteLog "Milvus containers started successfully">> "%VBS_SCRIPT%"
echo         Else>> "%VBS_SCRIPT%"
echo             WriteLog "Container startup returned code: " ^& result>> "%VBS_SCRIPT%"
echo         End If>> "%VBS_SCRIPT%"
echo     Else>> "%VBS_SCRIPT%"
echo         WriteLog "WARNING: Compose file not found: " ^& composeFile>> "%VBS_SCRIPT%"
echo         WriteLog "Skipping container startup">> "%VBS_SCRIPT%"
echo     End If>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     ' Create completion flag>> "%VBS_SCRIPT%"
echo     Dim flagFile>> "%VBS_SCRIPT%"
echo     Set flagFile = fso.CreateTextFile(startupComplete, True)>> "%VBS_SCRIPT%"
echo     flagFile.WriteLine "Startup completed at: " ^& Now>> "%VBS_SCRIPT%"
echo     flagFile.Close>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo     WriteLog "==========================================">> "%VBS_SCRIPT%"
echo     WriteLog "Podman Auto-Startup Script Completed">> "%VBS_SCRIPT%"
echo     WriteLog "==========================================">> "%VBS_SCRIPT%"
echo End Sub>> "%VBS_SCRIPT%"
echo.>> "%VBS_SCRIPT%"
echo ' Start the main process>> "%VBS_SCRIPT%"
echo Main()>> "%VBS_SCRIPT%"

if exist "%VBS_SCRIPT%" (
    echo [SUCCESS] VBS script created: %VBS_SCRIPT%
) else (
    echo [ERROR] Failed to create VBS script
    pause
    exit /b 1
)

REM Step 3: Register with Windows Task Scheduler
echo.
echo ================================================================
echo Step 3: Registering with Windows Task Scheduler
echo ================================================================

set "TASK_NAME=PodmanAutoStartup"
set "TASK_DESCRIPTION=Automatically start Podman and Milvus containers at Windows startup"

echo Creating scheduled task: %TASK_NAME%
echo.

REM Create task using schtasks command
schtasks /create /tn "%TASK_NAME%" /tr "wscript.exe \"%VBS_SCRIPT%\"" /sc onstart /ru "SYSTEM" /rl highest /f /st 00:00 /sd 01/01/2024

if %errorlevel% == 0 (
    echo [SUCCESS] Scheduled task created successfully!
    echo.
    echo Task Details:
    echo - Name: %TASK_NAME%
    echo - Trigger: At system startup
    echo - Action: Run VBS script silently
    echo - User: SYSTEM (highest privileges)
    echo - Script: %VBS_SCRIPT%
) else (
    echo [ERROR] Failed to create scheduled task
    echo.
    echo This might be due to insufficient privileges.
    echo Please run this script as Administrator.
    pause
    exit /b 1
)

REM Step 4: Verify setup
echo.
echo ================================================================
echo Step 4: Verifying setup
echo ================================================================

echo Checking if task was created...
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Task verification passed
    
    echo.
    echo Displaying task information:
    schtasks /query /tn "%TASK_NAME%" /fo list /v | findstr /i "TaskName: Status: Next Run Time:"
) else (
    echo [ERROR] Task verification failed
    echo The scheduled task was not created properly
    pause
    exit /b 1
)

REM Step 5: Test the VBS script
echo.
echo ================================================================
echo Step 5: Testing VBS script (optional)
echo ================================================================

set /p "test_choice=Do you want to test the VBS script now? (y/n): "
if /i "%test_choice%"=="y" (
    echo.
    echo Testing VBS script...
    echo This will run the startup script once to verify it works.
    echo Check the log file after completion: %PROJECT_DIR%\podman_startup.log
    echo.
    
    cscript //NoLogo "%VBS_SCRIPT%"
    
    echo.
    echo Test completed. Check the log file for details:
    if exist "%PROJECT_DIR%\podman_startup.log" (
        echo ----------------------------------------
        type "%PROJECT_DIR%\podman_startup.log"
        echo ----------------------------------------
    ) else (
        echo [WARNING] Log file not found
    )
)

REM Final summary
echo.
echo ================================================================
echo                    SETUP COMPLETE!
echo ================================================================
echo.
echo Auto-startup has been configured successfully:
echo.
echo [PODMAN PATH]
echo   %PODMAN_PATH%
echo.
echo [VBS SCRIPT]
echo   %VBS_SCRIPT%
echo.
echo [SCHEDULED TASK]
echo   Name: %TASK_NAME%
echo   Trigger: At Windows startup
echo   User: SYSTEM
echo.
echo [LOG FILES]
echo   Startup log: %PROJECT_DIR%\podman_startup.log
echo   Completion flag: %PROJECT_DIR%\startup_complete.flag
echo.
echo [WHAT HAPPENS AT STARTUP]
echo   1. Windows starts the scheduled task
echo   2. VBS script runs silently in background
echo   3. Podman machine starts (if needed)
echo   4. Milvus containers start automatically
echo   5. Log files are created for monitoring
echo.
echo [MANAGEMENT]
echo   - Enable task:  schtasks /change /tn "%TASK_NAME%" /enable
echo   - Disable task: schtasks /change /tn "%TASK_NAME%" /disable
echo   - Delete task:  schtasks /delete /tn "%TASK_NAME%" /f
echo   - View logs:    type "%PROJECT_DIR%\podman_startup.log"
echo.
echo [NEXT STEPS]
echo   1. Restart your computer to test auto-startup
echo   2. Check log files after restart
echo   3. Verify Milvus is running: http://localhost:19530
echo.
echo Your Podman and Milvus will now start automatically on every boot!
echo ================================================================

pause
exit /b 0
