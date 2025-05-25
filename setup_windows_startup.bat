@echo off
echo ================================================================
echo          Milvus Safe Startup - Windows Startup Setup
echo ================================================================
echo.
echo This script will help you set up automatic Milvus startup
echo that preserves all your embedding data safely.
echo.

REM Get current directory
set "CURRENT_DIR=%~dp0"
set "VBS_SCRIPT=%CURRENT_DIR%safe_start_milvus.vbs"

echo Current project directory: %CURRENT_DIR%
echo VBS Script location: %VBS_SCRIPT%
echo.

REM Check if VBS script exists
if not exist "%VBS_SCRIPT%" (
    echo ERROR: safe_start_milvus.vbs not found!
    echo Please make sure the VBS script is in the same directory.
    pause
    exit /b 1
)

echo ================================================================
echo                    SETUP OPTIONS
echo ================================================================
echo.
echo 1. Add to Windows Startup (Current User)
echo 2. Add to Windows Startup (All Users) - Requires Admin
echo 3. Create Desktop Shortcut Only
echo 4. Test Safe Startup Script
echo 5. Remove from Startup
echo 0. Exit
echo.

set /p choice="Select option (0-5): "

if "%choice%"=="1" goto :add_user_startup
if "%choice%"=="2" goto :add_all_startup
if "%choice%"=="3" goto :create_shortcut
if "%choice%"=="4" goto :test_script
if "%choice%"=="5" goto :remove_startup
if "%choice%"=="0" goto :exit
goto :invalid_choice

:add_user_startup
echo.
echo Adding to Windows Startup (Current User)...
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_DIR%\Milvus Safe Start.lnk"

REM Create shortcut using PowerShell
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%VBS_SCRIPT%'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Description = 'Safe Milvus Server Startup - Preserves All Data'; $Shortcut.Save()}"

if exist "%SHORTCUT_PATH%" (
    echo ✅ Successfully added to Windows Startup!
    echo Location: %SHORTCUT_PATH%
    echo.
    echo Milvus will now start safely every time you log in to Windows.
    echo Your embedding data will always be preserved.
) else (
    echo ❌ Failed to create startup shortcut.
    echo Please try running as administrator.
)
goto :finish

:add_all_startup
echo.
echo Adding to Windows Startup (All Users)...
echo This requires administrator privileges.
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Administrator privileges required for All Users startup.
    echo Please run this script as administrator or use option 1.
    goto :finish
)

set "STARTUP_DIR=%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_DIR%\Milvus Safe Start.lnk"

REM Create shortcut using PowerShell
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%VBS_SCRIPT%'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Description = 'Safe Milvus Server Startup - Preserves All Data'; $Shortcut.Save()}"

if exist "%SHORTCUT_PATH%" (
    echo ✅ Successfully added to Windows Startup for all users!
    echo Location: %SHORTCUT_PATH%
    echo.
    echo Milvus will now start safely for all users who log in.
) else (
    echo ❌ Failed to create startup shortcut.
)
goto :finish

:create_shortcut
echo.
echo Creating desktop shortcut only...
set "DESKTOP_DIR=%USERPROFILE%\Desktop"
set "SHORTCUT_PATH=%DESKTOP_DIR%\Milvus Safe Start.lnk"

powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%VBS_SCRIPT%'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Description = 'Safe Milvus Server Startup - Preserves All Data'; $Shortcut.Save()}"

if exist "%SHORTCUT_PATH%" (
    echo ✅ Desktop shortcut created successfully!
    echo Location: %SHORTCUT_PATH%
    echo.
    echo You can double-click this shortcut to start Milvus safely.
) else (
    echo ❌ Failed to create desktop shortcut.
)
goto :finish

:test_script
echo.
echo Testing safe startup script...
echo ================================================
echo Running: %VBS_SCRIPT%
echo ================================================
echo.

REM Run the VBS script
cscript //NoLogo "%VBS_SCRIPT%"

echo.
echo ================================================
echo Test completed. Check the log files for details:
echo - milvus_startup.log
echo - data_status.txt
echo ================================================
goto :finish

:remove_startup
echo.
echo Removing from Windows Startup...

REM Remove from current user startup
set "USER_SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Milvus Safe Start.lnk"
if exist "%USER_SHORTCUT%" (
    del "%USER_SHORTCUT%"
    echo ✅ Removed from current user startup
) else (
    echo ⚪ Not found in current user startup
)

REM Remove from all users startup (if admin)
set "ALL_SHORTCUT=%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Milvus Safe Start.lnk"
if exist "%ALL_SHORTCUT%" (
    del "%ALL_SHORTCUT%" 2>nul
    if not exist "%ALL_SHORTCUT%" (
        echo ✅ Removed from all users startup
    ) else (
        echo ⚠️ Could not remove from all users startup (admin required)
    )
) else (
    echo ⚪ Not found in all users startup
)

echo.
echo Milvus automatic startup has been disabled.
goto :finish

:invalid_choice
echo.
echo ❌ Invalid choice. Please select 0-5.
echo.
pause
goto :start

:finish
echo.
echo ================================================================
echo                     SETUP COMPLETE
echo ================================================================
echo.
echo 📋 What happens during safe startup:
echo ✅ Preserves ALL embedding data (volumes\etcd\, MilvusData\)
echo ✅ Only restarts containers safely
echo ✅ Waits for system to be ready
echo ✅ Creates detailed logs
echo ✅ Shows success/error notifications
echo.
echo 📁 Log files created:
echo   - milvus_startup.log (detailed startup log)
echo   - data_status.txt (data safety verification)
echo.
echo 💡 Your embedding data is 100%% safe with this startup method!
echo ================================================================
pause

:exit
echo.
echo Exiting setup...
exit /b 0
