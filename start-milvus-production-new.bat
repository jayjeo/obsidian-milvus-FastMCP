@echo off
echo Starting Milvus with Podman (Production - Persistent Storage)...

REM Ensure Podman machine is running
echo Starting Podman machine...

REM Try to start Podman machine
podman machine start

REM Check if Podman is responding
podman ps >nul 2>&1
if %errorlevel% neq 0 (
    echo Podman connection error detected. Attempting to restart Podman machine...
    echo Stopping Podman machine...
    podman machine stop
    timeout /t 5 /nobreak >nul
    echo Restarting Podman machine...
    podman machine start
    timeout /t 10 /nobreak >nul
    
    REM Check again if Podman is responding after restart
    podman ps >nul 2>&1
    if %errorlevel% neq 0 (
        echo Podman still not responding. Trying to reinitialize...
        podman machine rm -f podman-machine-default
        podman machine init
        podman machine start
        timeout /t 15 /nobreak >nul
    )
)

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
    exit /b 1
)

echo [OK] Podman machine is running and responding.

REM Stop and remove existing containers
echo Cleaning up existing containers...
podman stop milvus-etcd milvus-minio milvus-standalone 2>nul
podman rm milvus-etcd milvus-minio milvus-standalone 2>nul

REM Create network if not exists
echo Creating network...
podman network exists milvus || podman network create milvus

REM Create named volumes for persistent storage
echo Creating persistent volumes...
podman volume exists milvus-etcd-data || podman volume create milvus-etcd-data
podman volume exists milvus-minio-data || podman volume create milvus-minio-data  
podman volume exists milvus-db-data || podman volume create milvus-db-data

echo Starting etcd with persistent storage...
podman run -d --name milvus-etcd --network milvus ^
  -v milvus-etcd-data:/etcd ^
  -e ETCD_AUTO_COMPACTION_MODE=revision ^
  -e ETCD_AUTO_COMPACTION_RETENTION=1000 ^
  -e ETCD_QUOTA_BACKEND_BYTES=4294967296 ^
  --user 0:0 ^
  quay.io/coreos/etcd:v3.5.0 ^
  etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

echo Starting minio with persistent storage...
podman run -d --name milvus-minio --network milvus ^
  -v milvus-minio-data:/minio_data ^
  -e MINIO_ACCESS_KEY=minioadmin ^
  -e MINIO_SECRET_KEY=minioadmin ^
  --user 0:0 ^
  minio/minio:RELEASE.2023-03-20T20-16-18Z ^
  server /minio_data

echo Waiting for dependencies to start...
timeout /t 15 /nobreak >nul

echo Starting milvus with persistent storage...
podman run -d --name milvus-standalone --network milvus ^
  -v milvus-db-data:/var/lib/milvus ^
  -p 19530:19530 ^
  -p 9091:9091 ^
  -e ETCD_ENDPOINTS=milvus-etcd:2379 ^
  -e MINIO_ADDRESS=milvus-minio:9000 ^
  --user 0:0 ^
  milvusdb/milvus:v2.3.4 ^
  milvus run standalone

echo.
echo Waiting for services to start...
timeout /t 20 /nobreak >nul

echo.
echo Checking container status...
podman ps

echo.
echo Checking logs for any issues...
echo === ETCD Status ===
podman logs --tail 5 milvus-etcd
echo.
echo === MinIO Status ===  
podman logs --tail 5 milvus-minio
echo.
echo === Milvus Status ===
podman logs --tail 5 milvus-standalone

echo.
echo ================================================
echo Milvus Production Setup Complete!
echo ================================================
echo   - Milvus API: http://localhost:19530
echo   - Milvus Web UI: http://localhost:9091
echo   - Data is now persistent across restarts
echo.
echo Volume Information:
podman volume ls | findstr milvus
echo.
echo To stop all services: stop-milvus.bat
echo To backup data: backup-milvus.bat
echo ================================================

pause
