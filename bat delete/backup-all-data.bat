@echo off
echo ================================================================
echo             Complete Milvus Data Backup
echo        모든 embedding 데이터를 안전하게 백업합니다
echo ================================================================
echo.

REM Create backup directory with timestamp
set "timestamp=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"
set "backup_dir=Backup_%timestamp%"

echo Creating backup directory: %backup_dir%
mkdir "%backup_dir%" 2>nul

echo.
echo [1/4] Backing up MinIO data (실제 embedding 벡터)...
if exist "MilvusData\minio" (
    echo   📦 Copying MilvusData\minio to %backup_dir%\minio...
    xcopy "MilvusData\minio" "%backup_dir%\minio" /E /I /H /Y >nul
    echo   ✅ MinIO backup completed
) else (
    echo   ⚠️  MilvusData\minio not found
)

echo.
echo [2/4] Backing up Etcd metadata (스키마 및 메타데이터)...
if exist "volumes\etcd" (
    echo   📋 Copying volumes\etcd to %backup_dir%\etcd...
    xcopy "volumes\etcd" "%backup_dir%\etcd" /E /I /H /Y >nul
    echo   ✅ Etcd backup completed
) else (
    echo   ⚠️  volumes\etcd not found
)

echo.
echo [3/4] Backing up Milvus data (인덱스 및 캐시)...
if exist "MilvusData\milvus" (
    echo   🔍 Copying MilvusData\milvus to %backup_dir%\milvus...
    xcopy "MilvusData\milvus" "%backup_dir%\milvus" /E /I /H /Y >nul
    echo   ✅ Milvus backup completed
) else (
    echo   ⚠️  MilvusData\milvus not found
)

echo.
echo [4/4] Creating backup info file...
echo Backup created: %date% %time% > "%backup_dir%\backup_info.txt"
echo Project path: %CD% >> "%backup_dir%\backup_info.txt"
echo. >> "%backup_dir%\backup_info.txt"
echo Backup contents: >> "%backup_dir%\backup_info.txt"
echo   - minio/     : MinIO object storage (actual embedding vectors) >> "%backup_dir%\backup_info.txt"
echo   - etcd/      : Etcd metadata (collection schemas, index info) >> "%backup_dir%\backup_info.txt"
echo   - milvus/    : Milvus index cache and temporary files >> "%backup_dir%\backup_info.txt"

echo.
echo ================================================================
echo               Backup completed successfully!
echo ================================================================
echo.
echo 📁 Backup location: %CD%\%backup_dir%
echo.
echo 📦 Backup includes:
echo   ✅ MinIO data     (실제 embedding 벡터)
echo   ✅ Etcd metadata  (컬렉션 스키마, 인덱스 정보)  
echo   ✅ Milvus cache   (인덱스 캐시)
echo.
echo 🔄 To restore: use restore-backup.bat [backup_folder]
echo.

pause
