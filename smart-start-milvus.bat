@echo off
setlocal enabledelayedexpansion
title Smart Milvus Startup (Handles Compose Issues)

echo ================================================================
echo               Smart Milvus Startup Tool
echo ================================================================
echo This script will try multiple methods to start Milvus containers
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PODMAN_PATH=C:\Program Files\RedHat\Podman\podman.exe"
set "COMPOSE_FILE=milvus-podman-compose.yml"

echo Checking Podman installation...
if not exist "%PODMAN_PATH%" (
    echo ERROR: Podman not found at %PODMAN_PATH%
    echo Please install Podman Desktop
    pause
    exit /b 1
)

echo Podman found: %PODMAN_PATH%
echo.

echo Checking Podman machine status...
"%PODMAN_PATH%" machine list
echo.

echo Ensuring Podman machine is started...
"%PODMAN_PATH%" machine start podman-machine-default >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo Attempting to start Milvus containers...
echo ========================================

:: Method 1: Try built-in podman compose
echo Method 1: Using built-in podman compose...
"%PODMAN_PATH%" compose -f "%COMPOSE_FILE%" up -d
if not errorlevel 1 (
    echo SUCCESS: Containers started with built-in compose
    goto check_containers
)

echo Built-in compose failed, trying method 2...
echo.

:: Method 2: Try podman-compose
echo Method 2: Using podman-compose...
podman-compose -f "%COMPOSE_FILE%" up -d
if not errorlevel 1 (
    echo SUCCESS: Containers started with podman-compose
    goto check_containers
)

echo podman-compose failed, trying method 3...
echo.

:: Method 3: Manual container startup
echo Method 3: Starting containers individually...
echo.

echo Creating network...
"%PODMAN_PATH%" network create milvus 2>nul

echo Starting etcd container...
"%PODMAN_PATH%" run -d --name milvus-etcd ^
    --network milvus ^
    -v "%SCRIPT_DIR%\MilvusData\etcd:/etcd" ^
    -e ETCD_AUTO_COMPACTION_MODE=revision ^
    -e ETCD_AUTO_COMPACTION_RETENTION=1000 ^
    -e ETCD_QUOTA_BACKEND_BYTES=4294967296 ^
    --user 0:0 ^
    quay.io/coreos/etcd:v3.5.0 ^
    etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

if errorlevel 1 (
    echo Removing existing etcd container and retrying...
    "%PODMAN_PATH%" rm -f milvus-etcd 2>nul
    "%PODMAN_PATH%" run -d --name milvus-etcd ^
        --network milvus ^
        -v "%SCRIPT_DIR%\MilvusData\etcd:/etcd" ^
        -e ETCD_AUTO_COMPACTION_MODE=revision ^
        -e ETCD_AUTO_COMPACTION_RETENTION=1000 ^
        -e ETCD_QUOTA_BACKEND_BYTES=4294967296 ^
        --user 0:0 ^
        quay.io/coreos/etcd:v3.5.0 ^
        etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
)

echo Starting MinIO container...
"%PODMAN_PATH%" run -d --name milvus-minio ^
    --network milvus ^
    -v "%SCRIPT_DIR%\MilvusData\minio:/minio_data" ^
    -e MINIO_ACCESS_KEY=minioadmin ^
    -e MINIO_SECRET_KEY=minioadmin ^
    --user 0:0 ^
    minio/minio:RELEASE.2023-03-20T20-16-18Z ^
    server /minio_data

if errorlevel 1 (
    echo Removing existing minio container and retrying...
    "%PODMAN_PATH%" rm -f milvus-minio 2>nul
    "%PODMAN_PATH%" run -d --name milvus-minio ^
        --network milvus ^
        -v "%SCRIPT_DIR%\MilvusData\minio:/minio_data" ^
        -e MINIO_ACCESS_KEY=minioadmin ^
        -e MINIO_SECRET_KEY=minioadmin ^
        --user 0:0 ^
        minio/minio:RELEASE.2023-03-20T20-16-18Z ^
        server /minio_data
)

echo Waiting for dependencies to start...
timeout /t 10 /nobreak >nul

echo Starting Milvus standalone container...
"%PODMAN_PATH%" run -d --name milvus-standalone ^
    --network milvus ^
    -p 19530:19530 -p 9091:9091 ^
    -v "%SCRIPT_DIR%\MilvusData\milvus:/var/lib/milvus" ^
    -e ETCD_ENDPOINTS=milvus-etcd:2379 ^
    -e MINIO_ADDRESS=milvus-minio:9000 ^
    --user 0:0 ^
    milvusdb/milvus:v2.3.4 ^
    milvus run standalone

if errorlevel 1 (
    echo Removing existing milvus container and retrying...
    "%PODMAN_PATH%" rm -f milvus-standalone 2>nul
    "%PODMAN_PATH%" run -d --name milvus-standalone ^
        --network milvus ^
        -p 19530:19530 -p 9091:9091 ^
        -v "%SCRIPT_DIR%\MilvusData\milvus:/var/lib/milvus" ^
        -e ETCD_ENDPOINTS=milvus-etcd:2379 ^
        -e MINIO_ADDRESS=milvus-minio:9000 ^
        --user 0:0 ^
        milvusdb/milvus:v2.3.4 ^
        milvus run standalone
)

:check_containers
echo.
echo Checking container status...
echo ===========================
"%PODMAN_PATH%" ps --filter name=milvus

echo.
echo Waiting for Milvus to be ready...
echo =================================
timeout /t 15 /nobreak

:: Test connection
echo Testing Milvus connection on port 19530...
netstat -an | findstr :19530 >nul
if not errorlevel 1 (
    echo SUCCESS: Milvus appears to be running on port 19530
) else (
    echo WARNING: Port 19530 not detected, Milvus may still be starting
    echo Please wait a few more minutes and check again
)

echo.
echo ================================================================
echo                         SUMMARY
echo ================================================================
echo Milvus startup attempt completed.
echo.
echo To verify everything is working:
echo 1. Check containers: %PODMAN_PATH% ps
echo 2. Check logs: %PODMAN_PATH% logs milvus-standalone
echo 3. Test connection: python test_milvus.py
echo.
echo If issues persist:
echo 1. Run fix-podman-compose.bat first
echo 2. Check the log files in this directory
echo 3. Try complete-podman-reset.bat to start fresh
echo.

pause
