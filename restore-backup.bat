@echo off
echo ================================================================
echo             Complete Milvus Data Restore
echo        백업된 모든 embedding 데이터를 복원합니다
echo ================================================================
echo.

if "%1"=="" (
    echo ERROR: Please specify backup folder
    echo Usage: restore-backup.bat [backup_folder_name]
    echo.
    echo Available backups:
    dir /b Backup_* 2>nul
    echo.
    pause
    exit /b 1
)

set "backup_dir=%1"

if not exist "%backup_dir%" (
    echo ERROR: Backup folder '%backup_dir%' not found
    echo.
    pause
    exit /b 1
)

echo Restoring from: %backup_dir%
echo.
echo WARNING: This will overwrite current Milvus data!
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/5] Stopping Milvus containers...
podman stop milvus-standalone milvus-minio milvus-etcd 2>nul

echo.
echo [2/5] Removing current data directories...
if exist "MilvusData\minio" rmdir /s /q "MilvusData\minio" 2>nul
if exist "MilvusData\milvus" rmdir /s /q "MilvusData\milvus" 2>nul  
if exist "volumes\etcd" rmdir /s /q "volumes\etcd" 2>nul

echo.
echo [3/5] Restoring MinIO data...
if exist "%backup_dir%\minio" (
    echo   📦 Restoring MinIO data...
    xcopy "%backup_dir%\minio" "MilvusData\minio" /E /I /H /Y >nul
    echo   ✅ MinIO restored
) else (
    echo   ⚠️  MinIO backup not found in %backup_dir%
)

echo.
echo [4/5] Restoring Etcd metadata...
if exist "%backup_dir%\etcd" (
    echo   📋 Restoring Etcd metadata...
    mkdir "volumes" 2>nul
    xcopy "%backup_dir%\etcd" "volumes\etcd" /E /I /H /Y >nul
    echo   ✅ Etcd restored
) else (
    echo   ⚠️  Etcd backup not found in %backup_dir%
)

echo.
echo [5/5] Restoring Milvus data...
if exist "%backup_dir%\milvus" (
    echo   🔍 Restoring Milvus data...
    xcopy "%backup_dir%\milvus" "MilvusData\milvus" /E /I /H /Y >nul
    echo   ✅ Milvus restored
) else (
    echo   ⚠️  Milvus backup not found in %backup_dir%
)

echo.
echo ================================================================
echo               Restore completed successfully!
echo ================================================================
echo.
echo 🔄 Starting Milvus containers...
podman compose -f milvus-podman-compose.yml up -d

echo.
echo ✅ All embedding data restored from %backup_dir%
echo ✅ Containers are starting up...
echo.
echo Wait 30 seconds for services to initialize, then test with:
echo   python main.py
echo.

pause
