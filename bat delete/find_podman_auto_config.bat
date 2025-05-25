@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo              Podman Path Auto-Configuration
echo ================================================================
echo.
echo This script will find Podman and automatically configure it
echo for use with your Milvus project.
echo.

REM Check if we already have a config file
set "CONFIG_FILE=%~dp0podman_config.bat"
if exist "%CONFIG_FILE%" (
    echo Found existing configuration file: %CONFIG_FILE%
    echo.
    echo Current configuration:
    type "%CONFIG_FILE%"
    echo.
    set /p "overwrite=Overwrite existing configuration? (y/n): "
    if /i "!overwrite!" NEQ "y" (
        echo.
        echo Using existing configuration...
        call "%CONFIG_FILE%"
        if defined PODMAN_PATH (
            echo Podman configured at: !PODMAN_PATH!
            "!PODMAN_PATH!" --version
        )
        goto :end
    )
)

echo.
echo Searching for Podman installation...

set "PODMAN_FOUND="

REM Quick PATH check first
where podman.exe >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('where podman.exe 2^>nul') do (
        set "PODMAN_FOUND=%%i"
        echo [FOUND in PATH] %%i
        goto :found
    )
)

REM Check common installation directories
set "LOCATIONS[0]=%ProgramFiles%\RedHat\Podman\podman.exe"
set "LOCATIONS[1]=%ProgramFiles(x86)%\RedHat\Podman\podman.exe"
set "LOCATIONS[2]=%LOCALAPPDATA%\Podman\podman.exe"
set "LOCATIONS[3]=%ProgramFiles%\Podman\podman.exe"
set "LOCATIONS[4]=%ProgramFiles(x86)%\Podman\podman.exe"

for /L %%i in (0,1,4) do (
    if defined LOCATIONS[%%i] (
        if exist "!LOCATIONS[%%i]!" (
            set "PODMAN_FOUND=!LOCATIONS[%%i]!"
            echo [FOUND] !LOCATIONS[%%i]!
            goto :found
        )
    )
)

echo [NOT FOUND] Podman not found in common locations.
echo.
echo Please ensure Podman is installed:
echo   - Download from: https://podman.io/
echo   - Or install via package manager
echo.
echo After installation, run this script again.
goto :end

:found
echo.
echo Testing Podman installation...
"!PODMAN_FOUND!" --version >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Podman is working correctly!
    
    for /f "tokens=*" %%v in ('"!PODMAN_FOUND!" --version 2^>nul') do (
        echo Version: %%v
    )
    
    echo.
    echo Creating configuration file...
    
    REM Create configuration batch file
    echo @echo off > "%CONFIG_FILE%"
    echo REM Auto-generated Podman configuration >> "%CONFIG_FILE%"
    echo REM Generated on: %date% %time% >> "%CONFIG_FILE%"
    echo REM Podman location: !PODMAN_FOUND! >> "%CONFIG_FILE%"
    echo. >> "%CONFIG_FILE%"
    echo set "PODMAN_PATH=!PODMAN_FOUND!" >> "%CONFIG_FILE%"
    echo set "PODMAN_DIR=!PODMAN_FOUND!" >> "%CONFIG_FILE%"
    
    REM Remove filename to get directory
    for %%i in ("!PODMAN_FOUND!") do (
        echo set "PODMAN_DIR=%%~dpi" >> "%CONFIG_FILE%"
    )
    
    echo. >> "%CONFIG_FILE%"
    echo REM Add aliases for common commands >> "%CONFIG_FILE%"
    echo doskey podman="!PODMAN_FOUND!" $* >> "%CONFIG_FILE%"
    echo. >> "%CONFIG_FILE%"
    echo REM Verify configuration >> "%CONFIG_FILE%"
    echo if not exist "!PODMAN_FOUND!" ( >> "%CONFIG_FILE%"
    echo     echo [ERROR] Podman not found at configured path: !PODMAN_FOUND! >> "%CONFIG_FILE%"
    echo     echo Please run find_podman_auto_config.bat again >> "%CONFIG_FILE%"
    echo     exit /b 1 >> "%CONFIG_FILE%"
    echo ^) >> "%CONFIG_FILE%"
    
    echo [SUCCESS] Configuration saved to: %CONFIG_FILE%
    echo.
    echo ================================================================
    echo                    USAGE INSTRUCTIONS
    echo ================================================================
    echo.
    echo To use Podman in your scripts:
    echo.
    echo Method 1: Call the configuration file
    echo   call "%CONFIG_FILE%"
    echo   "%%PODMAN_PATH%%" --version
    echo.
    echo Method 2: Add to your script startup
    echo   Add this line to the beginning of your batch files:
    echo   if exist "%CONFIG_FILE%" call "%CONFIG_FILE%"
    echo.
    echo Method 3: Manual PATH addition
    
    for %%i in ("!PODMAN_FOUND!") do (
        echo   Add to PATH: %%~dpi
    )
    
    echo.
    echo ================================================================
    echo Creating example usage script...
    
    set "EXAMPLE_FILE=%~dp0test_podman_config.bat"
    echo @echo off > "%EXAMPLE_FILE%"
    echo echo Testing Podman configuration... >> "%EXAMPLE_FILE%"
    echo echo. >> "%EXAMPLE_FILE%"
    echo call "%CONFIG_FILE%" >> "%EXAMPLE_FILE%"
    echo echo Podman path: %%PODMAN_PATH%% >> "%EXAMPLE_FILE%"
    echo echo. >> "%EXAMPLE_FILE%"
    echo echo Running Podman version check: >> "%EXAMPLE_FILE%"
    echo "%%PODMAN_PATH%%" --version >> "%EXAMPLE_FILE%"
    echo echo. >> "%EXAMPLE_FILE%"
    echo echo Configuration test complete! >> "%EXAMPLE_FILE%"
    echo pause >> "%EXAMPLE_FILE%"
    
    echo [SUCCESS] Example script created: %EXAMPLE_FILE%
    echo.
    echo Run test_podman_config.bat to verify the configuration works.
    
) else (
    echo [ERROR] Podman found but not working properly.
    echo Please reinstall Podman or check your installation.
    echo.
    echo Podman location: !PODMAN_FOUND!
)

:end
echo.
echo ================================================================
echo Configuration complete!
echo ================================================================
pause
