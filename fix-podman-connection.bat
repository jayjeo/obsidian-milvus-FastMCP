@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo           Podman Connection Fixer
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check for administrator rights
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running as ADMINISTRATOR
) else (
    echo [INFO] Running as NORMAL USER
)
echo.

echo Current directory: %CD%
echo.

echo ================================================================
echo Step 1: Checking Podman Status
echo ================================================================

REM Find Podman executable
set "PODMAN_CMD="
if exist "C:\Program Files\RedHat\Podman\podman.exe" (
    set "PODMAN_CMD=C:\Program Files\RedHat\Podman\podman.exe"
) else (
    for /f "tokens=*" %%p in ('where podman 2^>nul') do (
        set "PODMAN_CMD=%%p"
        goto :found_podman
    )
)

:found_podman
if "%PODMAN_CMD%"=="" (
    echo ERROR: Podman not found
    pause
    exit /b 1
)

echo Using Podman: %PODMAN_CMD%
echo.

echo Checking Podman machine status...
"%PODMAN_CMD%" machine list
echo.

echo ================================================================
echo Step 2: Testing Podman Connection
echo ================================================================

echo Testing basic Podman connection...
"%PODMAN_CMD%" ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Podman connection working
    goto :check_containers
) else (
    echo [ERROR] Podman connection failed
    echo.
    echo This usually happens because:
    echo 1. WSL/VM connection issues when running as admin
    echo 2. Podman socket connection problems
    echo 3. Permission issues with Podman machine
    echo.
    echo Attempting to fix...
)

echo ================================================================
echo Step 3: Fixing Podman Connection
echo ================================================================

echo Method 1: Restarting Podman machine...
echo Stopping Podman machine...
"%PODMAN_CMD%" machine stop >nul 2>&1
timeout /t 5 /nobreak >nul

echo Starting Podman machine...
"%PODMAN_CMD%" machine start
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Podman machine
    goto :try_method2
)

timeout /t 10 /nobreak >nul

echo Testing connection after restart...
"%PODMAN_CMD%" ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Podman connection fixed!
    goto :check_containers
)

:try_method2
echo.
echo Method 2: Reinitializing Podman machine...
echo WARNING: This will remove existing Podman machine data
set /p "confirm=Continue with reinitialization? (y/N): "
if /i not "%confirm%"=="y" goto :manual_fix

echo Stopping and removing existing machine...
"%PODMAN_CMD%" machine stop >nul 2>&1
"%PODMAN_CMD%" machine rm -f podman-machine-default >nul 2>&1

echo Creating new machine...
"%PODMAN_CMD%" machine init
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize Podman machine
    goto :manual_fix
)

echo Starting new machine...
"%PODMAN_CMD%" machine start
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start new Podman machine
    goto :manual_fix
)

timeout /t 15 /nobreak >nul

echo Testing connection after reinitialization...
"%PODMAN_CMD%" ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Podman connection fixed after reinitialization!
    goto :check_containers
)

:manual_fix
echo.
echo ================================================================
echo Manual Fix Required
echo ================================================================
echo.
echo Automatic fixes failed. Please try these manual steps:
echo.
echo 1. Close all administrator command prompts
echo 2. Open a NORMAL user command prompt
echo 3. Run these commands:
echo    podman machine stop
echo    podman machine start
echo    podman ps
echo.
echo 4. If that doesn't work, try:
echo    podman machine rm -f podman-machine-default
echo    podman machine init
echo    podman machine start
echo.
echo 5. The issue might be that Podman WSL backend doesn't work well
echo    when accessed from administrator mode.
echo.
echo Alternative: Run the main program WITHOUT administrator privileges
echo.
goto :end

:check_containers
echo.
echo ================================================================
echo Step 4: Checking Milvus Containers
echo ================================================================

echo Current containers:
"%PODMAN_CMD%" ps -a
echo.

echo Checking if Milvus containers exist...
"%PODMAN_CMD%" ps -a --format "{{.Names}}" | findstr milvus >nul
if %errorlevel% equ 0 (
    echo [INFO] Milvus containers found
    
    echo Checking if Milvus containers are running...
    "%PODMAN_CMD%" ps --format "{{.Names}}" | findstr milvus >nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] Milvus containers are running
        goto :test_milvus
    ) else (
        echo [INFO] Milvus containers exist but not running
        echo Starting Milvus containers...
        
        "%PODMAN_CMD%" start milvus-etcd milvus-minio milvus-standalone >nul 2>&1
        if %errorlevel% equ 0 (
            echo [SUCCESS] Milvus containers started
            echo Waiting for services to initialize...
            timeout /t 30 /nobreak >nul
            goto :test_milvus
        ) else (
            echo [INFO] Could not start existing containers, will create new ones
        )
    )
) else (
    echo [INFO] No Milvus containers found
)

echo.
echo Starting fresh Milvus installation...
if exist "start-milvus.bat" (
    echo Running start-milvus.bat...
    call start-milvus.bat
) else (
    echo [ERROR] start-milvus.bat not found
    echo Please make sure you're in the project directory
    goto :end
)

:test_milvus
echo.
echo ================================================================
echo Step 5: Testing Milvus Connection
echo ================================================================

echo Waiting for Milvus to be ready...
timeout /t 15 /nobreak >nul

echo Testing Milvus connection...
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('[SUCCESS] Milvus is responding'); connections.disconnect('default')" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Milvus is working correctly!
    echo.
    echo You can now run main.py
) else (
    echo [INFO] Milvus not ready yet, this is normal
    echo Wait a few more minutes and try again
    echo.
    echo Or check the container logs:
    echo   %PODMAN_CMD% logs milvus-standalone
)

:end
echo.
echo ================================================================
echo Summary
echo ================================================================
echo.
echo If Podman connection issues persist when running as administrator:
echo.
echo RECOMMENDED SOLUTION: Run WITHOUT administrator privileges
echo   - Most MCP operations don't require admin rights
echo   - Podman works better with normal user privileges
echo   - Double-click run-main.bat normally (no right-click "Run as administrator")
echo.
echo If you specifically need admin rights for other reasons:
echo   - Try running from a normal command prompt first
echo   - Then elevate only when needed
echo.
pause
