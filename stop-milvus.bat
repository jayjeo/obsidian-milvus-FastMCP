@echo off
echo ================================================================
echo              Stopping Milvus Vector Database
echo ================================================================

REM Change to the script's directory
cd /d "%~dp0"

echo Stopping Milvus services...
podman stop milvus-standalone milvus-minio milvus-etcd 2>nul

echo Removing containers...
podman rm milvus-standalone milvus-minio milvus-etcd 2>nul

echo.
echo ✓ All Milvus containers stopped and removed
echo.
echo 💾 Your data is safe and preserved in volumes:
podman volume ls | findstr milvus
echo.
echo To restart Milvus: start-milvus.bat
echo To completely remove all data: remove-all-data.bat
echo.
echo ================================================================