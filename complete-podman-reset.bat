@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo           Complete Podman Reset and Fix
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo This script will completely reset and fix Podman connection issues.
echo.

set /p "confirm=Do you want to proceed with complete Podman reset? (y/N): "
if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo ================================================================
echo Step 1: Complete Podman Cleanup
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
    echo Please install Podman first
    pause
    exit /b 1
)

echo Using Podman: %PODMAN_CMD%
echo.

echo Stopping all containers and machines...
"%PODMAN_CMD%" stop --all --timeout 5 >nul 2>&1
"%PODMAN_CMD%" rm --all --force >nul 2>&1
"%PODMAN_CMD%" machine stop >nul 2>&1

echo Removing Podman machine completely...
"%PODMAN_CMD%" machine rm --force podman-machine-default >nul 2>&1

echo Cleaning up Podman system...
"%PODMAN_CMD%" system prune --all --force --volumes >nul 2>&1

echo ================================================================
echo Step 2: Recreating Podman Machine
echo ================================================================

echo Creating new Podman machine...
"%PODMAN_CMD%" machine init --memory 4096 --cpus 4 --disk-size 50
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize Podman machine
    echo Trying with default settings...
    "%PODMAN_CMD%" machine init
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create Podman machine with default settings
        goto :troubleshoot
    )
)

echo Starting Podman machine...
"%PODMAN_CMD%" machine start
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Podman machine
    goto :troubleshoot
)

echo ================================================================
echo Step 3: Testing Connection
echo ================================================================

echo Waiting for Podman to be ready...
timeout /t 15 /nobreak >nul

echo Testing basic connection...
for /l %%i in (1,1,10) do (
    "%PODMAN_CMD%" ps >nul 2>&1
    if !errorlevel! equ 0 (
        echo [SUCCESS] Podman connection working on attempt %%i
        goto :connection_success
    )
    echo Attempt %%i failed, retrying in 5 seconds...
    timeout /t 5 /nobreak >nul
)

echo [ERROR] Connection test failed after 10 attempts
goto :troubleshoot

:connection_success
echo ================================================================
echo Step 4: Starting Milvus Services
echo ================================================================

echo Podman is now working! Starting Milvus...

if exist "start-milvus.bat" (
    echo Running start-milvus.bat...
    call start-milvus.bat
    if %errorlevel% equ 0 (
        echo [SUCCESS] Milvus started successfully!
        goto :final_test
    ) else (
        echo [WARNING] start-milvus.bat had issues, but this might be normal
        echo Let's test if containers are running...
    )
) else (
    echo start-milvus.bat not found, trying manual start...
    
    echo Creating network...
    "%PODMAN_CMD%" network create milvus >nul 2>&1
    
    echo Starting etcd...
    "%PODMAN_CMD%" run -d --name milvus-etcd --network milvus -v "%CD%\volumes\etcd:/etcd" -e ETCD_AUTO_COMPACTION_MODE=revision -e ETCD_AUTO_COMPACTION_RETENTION=1000 -e ETCD_QUOTA_BACKEND_BYTES=4294967296 --user 0:0 quay.io/coreos/etcd:v3.5.0 etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    
    echo Starting MinIO...
    "%PODMAN_CMD%" run -d --name milvus-minio --network milvus -v "%CD%\MilvusData\minio:/minio_data" -e MINIO_ACCESS_KEY=minioadmin -e MINIO_SECRET_KEY=minioadmin --user 0:0 minio/minio:RELEASE.2023-03-20T20-16-18Z server /minio_data
    
    timeout /t 10 /nobreak >nul
    
    echo Starting Milvus...
    "%PODMAN_CMD%" run -d --name milvus-standalone --network milvus -v "%CD%\MilvusData\milvus:/var/lib/milvus" -p 19530:19530 -p 9091:9091 -e ETCD_ENDPOINTS=milvus-etcd:2379 -e MINIO_ADDRESS=milvus-minio:9000 --user 0:0 milvusdb/milvus:v2.3.4 milvus run standalone
)

:final_test
echo ================================================================
echo Step 5: Final Verification
echo ================================================================

echo Waiting for services to initialize...
timeout /t 30 /nobreak >nul

echo Checking container status:
"%PODMAN_CMD%" ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo Testing Milvus connection...
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('[SUCCESS] Milvus is responding!'); connections.disconnect('default')" 2>nul
if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo                    SUCCESS!
    echo ================================================================
    echo.
    echo ✅ Podman is working correctly
    echo ✅ Milvus is running and responding
    echo ✅ You can now run: run-main.bat
    echo.
) else (
    echo.
    echo [INFO] Milvus containers are starting but not ready yet.
    echo This is normal - Milvus can take 2-3 minutes to fully initialize.
    echo.
    echo Try running: run-main.bat in a few minutes.
)

goto :end

:troubleshoot
echo.
echo ================================================================
echo                    TROUBLESHOOTING
echo ================================================================
echo.
echo Podman machine creation/start failed. Common solutions:
echo.
echo 1. Restart your computer and try again
echo 2. Check if WSL2 is properly installed:
echo    wsl --status
echo    wsl --update
echo.
echo 3. Check if Hyper-V is enabled (for older Podman versions)
echo.
echo 4. Try running as administrator:
echo    Right-click this file and "Run as administrator"
echo.
echo 5. Check Windows version compatibility:
echo    Podman requires Windows 10 version 2004+ or Windows 11
echo.
echo 6. Alternative: Install Docker Desktop instead of Podman
echo    Download from: https://www.docker.com/products/docker-desktop/
echo.

:end
echo.
pause
