@echo off
echo ================================================================
echo              Complete Milvus Data Removal
echo ================================================================
echo.
echo WARNING: This will permanently delete ALL Milvus data!
echo This includes:
echo   • All vector collections
echo   • All indexes
echo   • All metadata
echo   • All configuration
echo.
set /p confirm="Are you sure? Type 'DELETE' to confirm: "
if not "%confirm%"=="DELETE" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo Stopping containers...
podman stop milvus-standalone milvus-minio milvus-etcd 2>nul
podman rm milvus-standalone milvus-minio milvus-etcd 2>nul

echo Removing network...
podman network rm milvus 2>nul

echo Removing ALL data volumes...
podman volume rm milvus-etcd-data milvus-minio-data milvus-db-data 2>nul

echo.
echo ✓ All Milvus data has been permanently deleted
echo.
echo To setup Milvus again: start-milvus.bat
echo.
echo ================================================================
pause