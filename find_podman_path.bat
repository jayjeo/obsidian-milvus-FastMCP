@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo                    Podman Path Finder
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Searching for podman.exe on your system...
echo This may take a moment depending on your system configuration.
echo.

set "PODMAN_FOUND=0"
set "PODMAN_PATHS="

echo ================================================================
echo 1. Checking PATH environment variable...
echo ================================================================

REM Check if podman is in PATH
where podman.exe >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Podman found in PATH!
    for /f "tokens=*" %%i in ('where podman.exe 2^>nul') do (
        echo   Location: %%i
        set "PODMAN_PATHS=!PODMAN_PATHS!%%i;"
        set "PODMAN_FOUND=1"
    )
    echo.
) else (
    echo [INFO] Podman not found in PATH environment variable.
    echo.
)

echo ================================================================
echo 2. Checking common installation directories...
echo ================================================================

REM Common Podman installation paths
set "COMMON_PATHS[0]=%ProgramFiles%\RedHat\Podman"
set "COMMON_PATHS[1]=%ProgramFiles(x86)%\RedHat\Podman"
set "COMMON_PATHS[2]=%ProgramFiles%\Podman"
set "COMMON_PATHS[3]=%ProgramFiles(x86)%\Podman"
set "COMMON_PATHS[4]=%LOCALAPPDATA%\Podman"
set "COMMON_PATHS[5]=%USERPROFILE%\AppData\Local\Podman"
set "COMMON_PATHS[6]=%USERPROFILE%\.local\bin"
set "COMMON_PATHS[7]=%USERPROFILE%\bin"
set "COMMON_PATHS[8]=%SYSTEMROOT%\System32"
set "COMMON_PATHS[9]=%SYSTEMROOT%\SysWOW64"
set "COMMON_PATHS[10]=C:\tools\podman"
set "COMMON_PATHS[11]=C:\bin"
set "COMMON_PATHS[12]=D:\tools\podman"
set "COMMON_PATHS[13]=E:\tools\podman"

for /L %%i in (0,1,13) do (
    if defined COMMON_PATHS[%%i] (
        set "CHECK_PATH=!COMMON_PATHS[%%i]!"
        if exist "!CHECK_PATH!\podman.exe" (
            echo [FOUND] !CHECK_PATH!\podman.exe
            set "PODMAN_PATHS=!PODMAN_PATHS!!CHECK_PATH!\podman.exe;"
            set "PODMAN_FOUND=1"
        ) else (
            echo [SKIP] !CHECK_PATH! - not found
        )
    )
)

echo.

echo ================================================================
echo 3. Searching entire C: drive (this may take several minutes)...
echo ================================================================
echo Please wait while we perform a comprehensive search...
echo.

REM Search C: drive for podman.exe
for /f "tokens=*" %%i in ('dir /s /b C:\podman.exe 2^>nul') do (
    echo [FOUND] %%i
    set "PODMAN_PATHS=!PODMAN_PATHS!%%i;"
    set "PODMAN_FOUND=1"
)

REM If C: search didn't find anything, check other drives
if !PODMAN_FOUND! == 0 (
    echo.
    echo ================================================================
    echo 4. Searching other drives...
    echo ================================================================
    
    for %%d in (D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
        if exist %%d:\ (
            echo Searching %%d: drive...
            for /f "tokens=*" %%i in ('dir /s /b %%d:\podman.exe 2^>nul') do (
                echo [FOUND] %%i
                set "PODMAN_PATHS=!PODMAN_PATHS!%%i;"
                set "PODMAN_FOUND=1"
            )
        )
    )
)

echo.
echo ================================================================
echo                        SEARCH RESULTS
echo ================================================================

if !PODMAN_FOUND! == 1 (
    echo [SUCCESS] Podman installations found:
    echo.
    
    REM Parse and display found paths
    set "COUNTER=1"
    for %%p in ("!PODMAN_PATHS:;=" "!") do (
        if "%%~p" NEQ "" (
            echo   !COUNTER!. %%~p
            set /a COUNTER+=1
        )
    )
    
    echo.
    echo ================================================================
    echo                    CONFIGURATION HELP
    echo ================================================================
    echo.
    echo To use Podman with your Milvus setup:
    echo.
    echo Option 1: Add to PATH Environment Variable
    echo   1. Copy one of the directory paths above (without podman.exe)
    echo   2. Add it to your system PATH environment variable
    echo   3. Restart your command prompt/PowerShell
    echo.
    echo Option 2: Use Full Path in Scripts
    echo   - Replace 'podman' with full path in your scripts
    echo   - Example: "C:\Program Files\RedHat\Podman\podman.exe"
    echo.
    echo Option 3: Create Symbolic Link (Advanced)
    echo   - Create a symlink in a directory that's already in PATH
    echo.
    
    REM Save results to file
    set "RESULT_FILE=%~dp0podman_paths.txt"
    echo Podman Path Search Results > "!RESULT_FILE!"
    echo Search Date: %date% %time% >> "!RESULT_FILE!"
    echo. >> "!RESULT_FILE!"
    echo Found Podman installations: >> "!RESULT_FILE!"
    
    set "COUNTER=1"
    for %%p in ("!PODMAN_PATHS:;=" "!") do (
        if "%%~p" NEQ "" (
            echo   !COUNTER!. %%~p >> "!RESULT_FILE!"
            set /a COUNTER+=1
        )
    )
    
    echo.
    echo Results saved to: !RESULT_FILE!
    
) else (
    echo [ERROR] Podman not found on this system!
    echo.
    echo Possible reasons:
    echo   1. Podman is not installed
    echo   2. Podman is installed in an unusual location
    echo   3. You don't have permission to access the installation directory
    echo.
    echo Solutions:
    echo   1. Install Podman Desktop from: https://podman.io/
    echo   2. Install Podman CLI from: https://github.com/containers/podman/releases
    echo   3. Use Docker instead of Podman (modify your compose files)
    echo.
    
    REM Save negative result to file
    set "RESULT_FILE=%~dp0podman_paths.txt"
    echo Podman Path Search Results > "!RESULT_FILE!"
    echo Search Date: %date% %time% >> "!RESULT_FILE!"
    echo. >> "!RESULT_FILE!"
    echo Result: Podman not found on this system >> "!RESULT_FILE!"
    echo. >> "!RESULT_FILE!"
    echo Please install Podman or check if it's installed in an unusual location. >> "!RESULT_FILE!"
)

echo.
echo ================================================================
echo                    ALTERNATIVE: DOCKER
echo ================================================================
echo.
echo If Podman is not available, you can use Docker instead:
echo.

REM Quick check for Docker
where docker.exe >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Docker is available on this system!
    for /f "tokens=*" %%i in ('where docker.exe 2^>nul') do (
        echo   Docker location: %%i
    )
    echo.
    echo You can use Docker Compose instead of Podman Compose.
    echo Just replace 'podman-compose' with 'docker-compose' in your scripts.
) else (
    echo [INFO] Docker not found in PATH.
    echo.
    echo To install Docker:
    echo   1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo   2. Or install Docker Engine for your platform
)

echo.
echo ================================================================
echo Search complete! Check podman_paths.txt for detailed results.
echo ================================================================
pause
