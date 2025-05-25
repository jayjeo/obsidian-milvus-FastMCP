@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo                Advanced Podman Path Finder
echo ================================================================
echo.
echo This script will:
echo   1. Find all Podman installations on your system
echo   2. Test each installation to verify it works
echo   3. Offer to configure the best one automatically
echo.

set "PODMAN_FOUND=0"
set "WORKING_PODMAN="
set "ALL_PODMAN_PATHS="

echo ================================================================
echo Phase 1: Quick PATH check
echo ================================================================

REM Check if podman is already in PATH and working
echo Checking if Podman is already configured in PATH...
where podman.exe >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Podman found in PATH!
    
    REM Test if it actually works
    echo Testing Podman functionality...
    podman --version >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=*" %%i in ('podman --version 2^>nul') do (
            echo [WORKING] %%i
            for /f "tokens=*" %%j in ('where podman.exe 2^>nul') do (
                set "WORKING_PODMAN=%%j"
                echo   Location: %%j
            )
        )
        set "PODMAN_FOUND=1"
        echo.
        echo Podman is already properly configured! No further action needed.
        goto :show_results
    ) else (
        echo [WARNING] Podman found in PATH but not working properly.
        echo Continuing search for working installation...
    )
    echo.
) else (
    echo [INFO] Podman not found in PATH. Searching system...
    echo.
)

echo ================================================================
echo Phase 2: Comprehensive system search
echo ================================================================

REM Define search locations with priority
set "SEARCH_LOCATIONS[0]=%ProgramFiles%\RedHat\Podman"
set "SEARCH_LOCATIONS[1]=%ProgramFiles(x86)%\RedHat\Podman"
set "SEARCH_LOCATIONS[2]=%LOCALAPPDATA%\Podman"
set "SEARCH_LOCATIONS[3]=%ProgramFiles%\Podman"
set "SEARCH_LOCATIONS[4]=%ProgramFiles(x86)%\Podman"
set "SEARCH_LOCATIONS[5]=%USERPROFILE%\AppData\Local\Podman"
set "SEARCH_LOCATIONS[6]=%USERPROFILE%\.local\bin"
set "SEARCH_LOCATIONS[7]=%USERPROFILE%\bin"
set "SEARCH_LOCATIONS[8]=C:\tools\podman"
set "SEARCH_LOCATIONS[9]=C:\bin"

echo Searching common installation directories...
for /L %%i in (0,1,9) do (
    if defined SEARCH_LOCATIONS[%%i] (
        set "CHECK_PATH=!SEARCH_LOCATIONS[%%i]!"
        echo Checking: !CHECK_PATH!
        
        if exist "!CHECK_PATH!\podman.exe" (
            echo   [FOUND] podman.exe detected
            
            REM Test if this installation works
            echo   [TEST] Verifying functionality...
            "!CHECK_PATH!\podman.exe" --version >nul 2>&1
            if !errorlevel! == 0 (
                for /f "tokens=*" %%v in ('"!CHECK_PATH!\podman.exe" --version 2^>nul') do (
                    echo   [WORKING] %%v
                    if "!WORKING_PODMAN!" == "" (
                        set "WORKING_PODMAN=!CHECK_PATH!\podman.exe"
                    )
                )
                set "ALL_PODMAN_PATHS=!ALL_PODMAN_PATHS!!CHECK_PATH!\podman.exe;"
                set "PODMAN_FOUND=1"
            ) else (
                echo   [BROKEN] Installation found but not working
                set "ALL_PODMAN_PATHS=!ALL_PODMAN_PATHS!!CHECK_PATH!\podman.exe (BROKEN);"
            )
        else (
            echo   [SKIP] Not found
        )
        echo.
    )
)

REM If no working installation found, do deep search
if "!WORKING_PODMAN!" == "" (
    echo ================================================================
    echo Phase 3: Deep system search (this may take a few minutes)
    echo ================================================================
    echo.
    echo Searching entire system for podman.exe...
    echo Please be patient, this comprehensive search may take time.
    echo.
    
    REM Search all drives
    for %%d in (C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
        if exist %%d:\ (
            echo Searching %%d: drive...
            for /f "tokens=*" %%i in ('dir /s /b "%%d:\podman.exe" 2^>nul') do (
                echo   [FOUND] %%i
                
                REM Test this installation
                echo   [TEST] Verifying...
                "%%i" --version >nul 2>&1
                if !errorlevel! == 0 (
                    for /f "tokens=*" %%v in ('"%%i" --version 2^>nul') do (
                        echo   [WORKING] %%v
                        if "!WORKING_PODMAN!" == "" (
                            set "WORKING_PODMAN=%%i"
                        )
                    )
                    set "ALL_PODMAN_PATHS=!ALL_PODMAN_PATHS!%%i;"
                    set "PODMAN_FOUND=1"
                ) else (
                    echo   [BROKEN] Found but not functional
                    set "ALL_PODMAN_PATHS=!ALL_PODMAN_PATHS!%%i (BROKEN);"
                )
                echo.
            )
        )
    )
)

:show_results
echo ================================================================
echo                        SEARCH RESULTS
echo ================================================================

if "!WORKING_PODMAN!" NEQ "" (
    echo [SUCCESS] Working Podman installation found!
    echo.
    echo Primary working installation:
    echo   !WORKING_PODMAN!
    
    REM Show version info
    for /f "tokens=*" %%v in ('"!WORKING_PODMAN!" --version 2^>nul') do (
        echo   Version: %%v
    )
    
    if "!ALL_PODMAN_PATHS!" NEQ "" (
        echo.
        echo All Podman installations found:
        set "COUNTER=1"
        for %%p in ("!ALL_PODMAN_PATHS:;=" "!") do (
            if "%%~p" NEQ "" (
                echo   !COUNTER!. %%~p
                set /a COUNTER+=1
            )
        )
    )
    
    echo.
    echo ================================================================
    echo                    CONFIGURATION OPTIONS
    echo ================================================================
    echo.
    echo Choose an option:
    echo   1. Add working Podman to PATH (Recommended)
    echo   2. Create batch file shortcuts
    echo   3. Show manual configuration instructions
    echo   4. Test Podman with a simple command
    echo   0. Exit without changes
    echo.
    
    set /p "choice=Enter your choice (0-4): "
    
    if "!choice!" == "1" goto :add_to_path
    if "!choice!" == "2" goto :create_shortcuts
    if "!choice!" == "3" goto :show_manual
    if "!choice!" == "4" goto :test_podman
    if "!choice!" == "0" goto :exit
    
    echo Invalid choice. Showing manual instructions...
    goto :show_manual

) else (
    echo [ERROR] No working Podman installation found!
    echo.
    
    if "!ALL_PODMAN_PATHS!" NEQ "" (
        echo Found non-working installations:
        for %%p in ("!ALL_PODMAN_PATHS:;=" "!") do (
            if "%%~p" NEQ "" (
                echo   - %%~p
            )
        )
        echo.
        echo These installations were found but are not functional.
        echo You may need to reinstall Podman.
    )
    
    echo ================================================================
    echo                    INSTALLATION GUIDANCE
    echo ================================================================
    echo.
    echo To install Podman:
    echo.
    echo Option 1: Podman Desktop (Recommended for beginners)
    echo   - Download from: https://podman.io/
    echo   - Includes GUI and CLI tools
    echo   - Easier setup and management
    echo.
    echo Option 2: Podman CLI only
    echo   - Download from: https://github.com/containers/podman/releases
    echo   - Lighter weight, command-line only
    echo   - For advanced users
    echo.
    echo Option 3: Use Docker instead
    echo   - Download Docker Desktop: https://www.docker.com/products/docker-desktop/
    echo   - Compatible with most Podman commands
    echo   - May require script modifications
    
    goto :check_docker
)

:add_to_path
echo.
echo ================================================================
echo                    Adding Podman to PATH
echo ================================================================

REM Extract directory from full path
for %%i in ("!WORKING_PODMAN!") do set "PODMAN_DIR=%%~dpi"
set "PODMAN_DIR=!PODMAN_DIR:~0,-1!"

echo.
echo Adding directory to PATH: !PODMAN_DIR!
echo.

REM Add to user PATH (safer than system PATH)
for /f "tokens=*" %%i in ('reg query "HKCU\Environment" /v PATH 2^>nul') do (
    set "CURRENT_PATH=%%i"
    set "CURRENT_PATH=!CURRENT_PATH:*REG_SZ=!"
    set "CURRENT_PATH=!CURRENT_PATH:*REG_EXPAND_SZ=!"
)

REM Check if already in PATH
echo !CURRENT_PATH! | findstr /i "!PODMAN_DIR!" >nul
if !errorlevel! == 0 (
    echo [INFO] Directory already in PATH!
) else (
    echo Adding to user PATH environment variable...
    reg add "HKCU\Environment" /v PATH /d "!CURRENT_PATH!;!PODMAN_DIR!" /f >nul
    if !errorlevel! == 0 (
        echo [SUCCESS] Added to PATH successfully!
        echo.
        echo IMPORTANT: You need to restart your command prompt or PowerShell
        echo for the changes to take effect.
    ) else (
        echo [ERROR] Failed to add to PATH. You may need administrator privileges.
    )
)

echo.
echo To verify the change worked:
echo   1. Close this window
echo   2. Open a new command prompt
echo   3. Type: podman --version
goto :save_results

:create_shortcuts
echo.
echo ================================================================
echo                    Creating Shortcuts
echo ================================================================

set "SHORTCUT_DIR=%~dp0"
set "PODMAN_BAT=!SHORTCUT_DIR!podman.bat"

echo Creating podman.bat shortcut...
echo @echo off > "!PODMAN_BAT!"
echo "!WORKING_PODMAN!" %%* >> "!PODMAN_BAT!"

if exist "!PODMAN_BAT!" (
    echo [SUCCESS] Created: !PODMAN_BAT!
    echo.
    echo You can now use 'podman.bat' instead of the full path.
    echo Or add this directory to your PATH: !SHORTCUT_DIR!
) else (
    echo [ERROR] Failed to create shortcut file.
)
goto :save_results

:show_manual
echo.
echo ================================================================
echo                  MANUAL CONFIGURATION
echo ================================================================
echo.
echo Working Podman found at: !WORKING_PODMAN!
echo.
echo Manual setup options:
echo.
echo 1. Add to PATH Environment Variable:
echo    - Open System Properties ^> Advanced ^> Environment Variables
echo    - Edit USER PATH variable
echo    - Add: !PODMAN_DIR!
echo    - Click OK and restart command prompt
echo.
echo 2. Use full path in scripts:
echo    - Replace 'podman' with: "!WORKING_PODMAN!"
echo    - Example: "!WORKING_PODMAN!" --version
echo.
echo 3. Create alias (PowerShell):
echo    - Add to PowerShell profile:
echo    - Set-Alias podman "!WORKING_PODMAN!"
goto :save_results

:test_podman
echo.
echo ================================================================
echo                      TESTING PODMAN
echo ================================================================
echo.
echo Running test commands with found Podman installation...
echo.

echo Test 1: Version check
"!WORKING_PODMAN!" --version
echo.

echo Test 2: System info
"!WORKING_PODMAN!" system info
echo.

echo Test 3: List images
"!WORKING_PODMAN!" images
echo.

echo ================================================================
echo Test completed. If you see errors, Podman may need additional setup.
goto :save_results

:check_docker
echo.
echo ================================================================
echo                    CHECKING FOR DOCKER
echo ================================================================

where docker.exe >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Docker is available as an alternative!
    for /f "tokens=*" %%i in ('where docker.exe 2^>nul') do (
        echo   Docker location: %%i
    )
    docker --version 2>nul
    echo.
    echo You can use Docker instead of Podman for container management.
    echo Most podman commands work with docker by simply replacing 'podman' with 'docker'.
) else (
    echo [INFO] Docker also not found on this system.
    echo Consider installing either Podman or Docker for container support.
)
goto :save_results

:save_results
echo.
echo ================================================================
echo                    SAVING RESULTS
echo ================================================================

set "RESULT_FILE=%~dp0podman_search_results.txt"
echo Podman Path Search Results > "!RESULT_FILE!"
echo Search Date: %date% %time% >> "!RESULT_FILE!"
echo Computer: %COMPUTERNAME% >> "!RESULT_FILE!"
echo User: %USERNAME% >> "!RESULT_FILE!"
echo. >> "!RESULT_FILE!"

if "!WORKING_PODMAN!" NEQ "" (
    echo WORKING PODMAN FOUND: >> "!RESULT_FILE!"
    echo   Primary: !WORKING_PODMAN! >> "!RESULT_FILE!"
    "!WORKING_PODMAN!" --version >> "!RESULT_FILE!" 2>&1
    echo. >> "!RESULT_FILE!"
    
    if "!ALL_PODMAN_PATHS!" NEQ "" (
        echo ALL INSTALLATIONS: >> "!RESULT_FILE!"
        for %%p in ("!ALL_PODMAN_PATHS:;=" "!") do (
            if "%%~p" NEQ "" (
                echo   - %%~p >> "!RESULT_FILE!"
            )
        )
        echo. >> "!RESULT_FILE!"
    )
) else (
    echo NO WORKING PODMAN FOUND >> "!RESULT_FILE!"
    echo. >> "!RESULT_FILE!"
    
    if "!ALL_PODMAN_PATHS!" NEQ "" (
        echo NON-WORKING INSTALLATIONS: >> "!RESULT_FILE!"
        for %%p in ("!ALL_PODMAN_PATHS:;=" "!") do (
            if "%%~p" NEQ "" (
                echo   - %%~p >> "!RESULT_FILE!"
            )
        )
        echo. >> "!RESULT_FILE!"
    )
)

echo RECOMMENDATIONS: >> "!RESULT_FILE!"
if "!WORKING_PODMAN!" NEQ "" (
    echo   - Add to PATH: !PODMAN_DIR! >> "!RESULT_FILE!"
    echo   - Or use full path: !WORKING_PODMAN! >> "!RESULT_FILE!"
) else (
    echo   - Install Podman from: https://podman.io/ >> "!RESULT_FILE!"
    echo   - Or install Docker as alternative >> "!RESULT_FILE!"
)

echo.
echo Results saved to: !RESULT_FILE!

:exit
echo.
echo ================================================================
echo                        SEARCH COMPLETE
echo ================================================================
echo.
echo Summary:
if "!WORKING_PODMAN!" NEQ "" (
    echo   [SUCCESS] Working Podman found and ready to use
    echo   Location: !WORKING_PODMAN!
) else (
    echo   [NOTICE] No working Podman installation found
    echo   Consider installing Podman or Docker for container support
)
echo.
echo Check podman_search_results.txt for detailed information.
echo.
pause
