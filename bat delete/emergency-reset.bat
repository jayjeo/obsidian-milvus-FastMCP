@echo off
echo ================================================================
echo           Emergency Milvus Reset Script
echo       This script will forcefully clean up everything
echo ================================================================
echo.

echo WARNING: This will completely destroy all Milvus data and containers!
echo Press Ctrl+C to cancel, or
pause

echo Starting emergency cleanup...

echo.
echo [1/4] Stopping all containers forcefully...
podman stop --all --timeout 5
podman kill --all
podman container prune --force

echo.
echo [2/4] Removing all Milvus containers...
podman rm --force milvus-standalone milvus-minio milvus-etcd
podman rm --force --all

echo.
echo [3/4] Cleaning up pods, volumes, and networks...
podman pod stop --all
podman pod rm --all --force
podman volume rm milvus-etcd-data milvus-minio-data milvus-db-data --force
podman network rm milvus milvus-network --force
podman system prune --all --force --volumes

echo.
echo [4/4] Removing local data directories...
if exist "MilvusData" (
    echo Removing MilvusData folder...
    rmdir /s /q "MilvusData"
)

if exist "volumes" (
    echo Removing volumes folder...
    rmdir /s /q "volumes"
)

echo.
echo ================================================================
echo            Emergency cleanup completed!
echo ================================================================
echo.
echo Now you can safely run "start-milvus.bat" to restart Milvus
echo or use option 2 in main.py for full embedding
echo.

pause
