@echo off
echo ================================================================
echo          Milvus Vector Database - Easy Setup Script
echo                    (Production Ready)
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check if Podman is installed
where podman >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Podman is not installed or not in PATH
    echo Please install Podman first:
    echo   winget install RedHat.Podman
    echo.
    pause
    exit /b 1
)

echo [OK] Podman found: 
podman --version

REM Check and start Podman machine
echo.
echo Starting Podman machine...

REM Check if Podman machine exists
podman machine list | findstr "podman-machine-default" >nul 2>&1
if %errorlevel% equ 0 (
    echo Podman machine already exists, starting it...
    podman machine start 2>nul
    if %errorlevel% neq 0 (
        echo ERROR: Failed to start existing Podman machine
        pause
        exit /b 1
    )
) else (
    echo First time setup - initializing Podman machine...
    podman machine init
    if %errorlevel% neq 0 (
        echo ERROR: Failed to initialize Podman machine
        pause
        exit /b 1
    )
    podman machine start
    if %errorlevel% neq 0 (
        echo ERROR: Failed to start Podman machine
        pause
        exit /b 1
    )
)

REM Check if Podman is responding (socket connection test)
podman ps >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Podman machine is running but socket connection failed.
    echo Attempting to fix Podman connection issues...
    
    echo Step 1: Stopping Podman machine...
    podman machine stop
    timeout /t 5 /nobreak >nul
    
    echo Step 2: Restarting Podman machine...
    podman machine start
    timeout /t 10 /nobreak >nul
    
    REM Check again if Podman is responding after restart
    podman ps >nul 2>&1
    if %errorlevel% neq 0 (
        echo Step 3: Connection still failed. Trying to reinitialize Podman machine...
        podman machine rm -f podman-machine-default
        podman machine init
        podman machine start
        timeout /t 15 /nobreak >nul
        
        REM Final check
        podman ps >nul 2>&1
        if %errorlevel% neq 0 (
            echo ERROR: Could not establish connection to Podman after multiple attempts.
            echo Please try the following steps manually:
            echo 1. podman machine stop
            echo 2. podman machine rm -f podman-machine-default
            echo 3. podman machine init
            echo 4. podman machine start
            echo 5. Run this script again
            pause
            exit /b 1
        )
    )
    echo [OK] Podman connection issues resolved
)

echo [OK] Podman machine is running and responding

REM Stop and remove existing containers
echo.
echo Cleaning up existing Milvus containers...
podman stop milvus-etcd milvus-minio milvus-standalone 2>nul
podman rm milvus-etcd milvus-minio milvus-standalone 2>nul

REM Create network if not exists
echo Creating Milvus network...
podman network exists milvus 2>nul || podman network create milvus
echo [OK] Network ready

REM Create local directories for persistent storage
echo.
echo Setting up persistent storage directories...
if not exist "volumes" mkdir "volumes"
if not exist "volumes\etcd" mkdir "volumes\etcd"
if not exist "MilvusData" mkdir "MilvusData"
if not exist "MilvusData\minio" mkdir "MilvusData\minio"
if not exist "MilvusData\milvus" mkdir "MilvusData\milvus"
echo [OK] Storage directories ready

echo.
echo ================================================================
echo                    Starting Milvus Services
echo ================================================================

echo [1/3] Starting etcd (metadata store)...
podman run -d --name milvus-etcd --network milvus ^
  -v "%CD%\volumes\etcd:/etcd" ^
  -e ETCD_AUTO_COMPACTION_MODE=revision ^
  -e ETCD_AUTO_COMPACTION_RETENTION=1000 ^
  -e ETCD_QUOTA_BACKEND_BYTES=4294967296 ^
  --user 0:0 ^
  quay.io/coreos/etcd:v3.5.0 ^
  etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

if %errorlevel% neq 0 (
    echo ERROR: Failed to start etcd
    pause
    exit /b 1
)
echo [OK] etcd started

echo [2/3] Starting MinIO (object storage)...
podman run -d --name milvus-minio --network milvus ^
  -v "%CD%\MilvusData\minio:/minio_data" ^
  -e MINIO_ACCESS_KEY=minioadmin ^
  -e MINIO_SECRET_KEY=minioadmin ^
  --user 0:0 ^
  minio/minio:RELEASE.2023-03-20T20-16-18Z ^
  server /minio_data

if %errorlevel% neq 0 (
    echo ERROR: Failed to start MinIO
    pause
    exit /b 1
)
echo [OK] MinIO started

echo Waiting for dependencies to initialize...
timeout /t 15 /nobreak >nul

echo [3/3] Starting Milvus (vector database)...
podman run -d --name milvus-standalone --network milvus ^
  -v "%CD%\MilvusData\milvus:/var/lib/milvus" ^
  -p 19530:19530 ^
  -p 9091:9091 ^
  -e ETCD_ENDPOINTS=milvus-etcd:2379 ^
  -e MINIO_ADDRESS=milvus-minio:9000 ^
  --user 0:0 ^
  milvusdb/milvus:v2.3.4 ^
  milvus run standalone

if %errorlevel% neq 0 (
    echo ERROR: Failed to start Milvus
    pause
    exit /b 1
)
echo [OK] Milvus started

echo.
echo Waiting for all services to be ready...
timeout /t 60 /nobreak >nul

echo.
echo ================================================================
echo                      Setup Complete!
echo ================================================================

echo Container Status:
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo Storage Directories:
echo   - Container data: %CD%\volumes
echo   - Embedding data: %CD%\MilvusData

echo.
echo ================================================================
echo                    SUCCESS!
echo ================================================================
echo Your Milvus vector database is now running!
echo.
echo Access URLs:
echo   - Milvus API:    http://localhost:19530
echo   - Web Interface: http://localhost:9091
echo.
echo Data Storage:
echo   - All data is persistent and will survive restarts
echo   - Container data: %CD%\volumes
echo   - Embedding data: %CD%\MilvusData
echo.

pause
