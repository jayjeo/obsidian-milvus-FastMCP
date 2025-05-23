@echo off
echo Starting Milvus with Podman (Fixed Permissions)...

REM Ensure Podman machine is running
podman machine start 2>nul

REM Stop and remove existing containers
echo Cleaning up existing containers...
podman stop milvus-etcd milvus-minio milvus-standalone 2>nul
podman rm milvus-etcd milvus-minio milvus-standalone 2>nul
podman network rm milvus 2>nul

REM Create network
echo Creating network...
podman network create milvus

REM Create directories (they will be created with proper permissions by containers)
echo Creating data directories...
if not exist "EmbeddingResult" mkdir "EmbeddingResult"

echo Starting etcd...
podman run -d --name milvus-etcd --network milvus ^
  --tmpfs /etcd:rw,exec,size=1g ^
  -e ETCD_AUTO_COMPACTION_MODE=revision ^
  -e ETCD_AUTO_COMPACTION_RETENTION=1000 ^
  -e ETCD_QUOTA_BACKEND_BYTES=4294967296 ^
  quay.io/coreos/etcd:v3.5.0 ^
  etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

echo Starting minio...
podman run -d --name milvus-minio --network milvus ^
  --tmpfs /minio_data:rw,exec,size=2g ^
  -e MINIO_ACCESS_KEY=minioadmin ^
  -e MINIO_SECRET_KEY=minioadmin ^
  minio/minio:RELEASE.2023-03-20T20-16-18Z ^
  server /minio_data

echo Waiting for dependencies to start...
timeout /t 10 /nobreak >nul

echo Starting milvus...
podman run -d --name milvus-standalone --network milvus ^
  --tmpfs /var/lib/milvus:rw,exec,size=2g ^
  -p 19530:19530 ^
  -p 9091:9091 ^
  -e ETCD_ENDPOINTS=milvus-etcd:2379 ^
  -e MINIO_ADDRESS=milvus-minio:9000 ^
  milvusdb/milvus:v2.3.4 ^
  milvus run standalone

echo.
echo Waiting for services to start...
timeout /t 15 /nobreak >nul

echo.
echo Checking container status...
podman ps

echo.
echo Milvus should be ready now. Check:
echo   - Milvus API: http://localhost:19530
echo   - Milvus Web UI: http://localhost:9091
echo.
echo Note: Using tmpfs (temporary storage) - data will not persist after restart
echo To stop all services, run: stop-milvus.bat
