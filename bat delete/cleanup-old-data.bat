@echo off
echo ================================================================
echo Cleanup Old Milvus Data Folders Script
echo ================================================================
echo.
echo WARNING: This script will DELETE G:\volumes and G:\MilvusData folders!
echo.
echo Only run this after migration is complete and Milvus works properly
echo in the new location.
echo.
echo Folders to be deleted:
echo - G:\volumes
echo - G:\MilvusData
echo.

set /p confirm1="Really delete the old folders? (y/N): "
if /i not "%confirm1%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 1
)

echo.
set /p confirm2="Final confirmation: Have you made a backup and verified Milvus works? (y/N): "
if /i not "%confirm2%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 1
)

echo.
echo Stopping Milvus containers...
podman compose -f milvus-podman-compose.yml down

echo.
echo Deleting old folders...

if exist "G:\volumes" (
    echo Deleting G:\volumes...
    rmdir /s /q "G:\volumes"
    if exist "G:\volumes" (
        echo [ERROR] Failed to delete G:\volumes
    ) else (
        echo [OK] G:\volumes deleted successfully
    )
)

if exist "G:\MilvusData" (
    echo Deleting G:\MilvusData...
    rmdir /s /q "G:\MilvusData"
    if exist "G:\MilvusData" (
        echo [ERROR] Failed to delete G:\MilvusData
    ) else (
        echo [OK] G:\MilvusData deleted successfully
    )
)

echo.
echo ================================================================
echo Cleanup completed!
echo ================================================================
echo.
echo All Milvus data is now in the project folder:
echo - etcd: volumes\etcd
echo - minio: MilvusData\minio
echo - milvus: MilvusData\milvus
echo.
pause
