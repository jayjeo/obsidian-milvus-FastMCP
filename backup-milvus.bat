@echo off
echo Backing up Milvus data...

REM Create backup directory with timestamp
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
set backup_dir=milvus_backup_%timestamp%

echo Creating backup directory: %backup_dir%
mkdir "%backup_dir%"

REM Export volume data
echo Backing up etcd data...
podman run --rm -v milvus-etcd-data:/source -v "%cd%\%backup_dir%:/backup" alpine tar czf /backup/etcd_data.tar.gz -C /source .

echo Backing up minio data...  
podman run --rm -v milvus-minio-data:/source -v "%cd%\%backup_dir%:/backup" alpine tar czf /backup/minio_data.tar.gz -C /source .

echo Backing up milvus data...
podman run --rm -v milvus-db-data:/source -v "%cd%\%backup_dir%:/backup" alpine tar czf /backup/milvus_data.tar.gz -C /source .

echo.
echo Backup completed: %backup_dir%
echo Files created:
dir "%backup_dir%"
