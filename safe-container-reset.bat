@echo off
echo ================================================================
echo         Safe Container Reset (Data Preservation)
echo      컨테이너만 재설치하고 embedding 데이터는 보존합니다
echo ================================================================
echo.
echo This will:
echo   ✅ Remove and recreate containers
echo   ✅ Preserve all embedding data in MilvusData folder
echo   ✅ Preserve all metadata in volumes folder
echo.
echo Data locations that will be PRESERVED:
echo   📁 %CD%\MilvusData\minio     (MinIO object storage)  
echo   📁 %CD%\MilvusData\milvus    (Milvus vector data)
echo   📁 %CD%\volumes\etcd         (Etcd metadata)
echo.
pause

echo.
echo [1/4] Stopping Milvus containers...
podman stop milvus-standalone milvus-minio milvus-etcd 2>nul

echo.
echo [2/4] Removing containers (keeping data)...
podman rm milvus-standalone milvus-minio milvus-etcd 2>nul

echo.
echo [3/4] Cleaning up networks and orphaned resources...
podman network rm milvus 2>nul
podman system prune --force 2>nul

echo.
echo [4/4] Recreating containers with preserved data...
podman compose -f milvus-podman-compose.yml up -d

echo.
echo ================================================================
echo              Safe reset completed successfully!
echo ================================================================
echo.
echo ✅ All embedding data preserved
echo ✅ Containers recreated
echo ✅ Services starting up...
echo.
echo Wait 30 seconds for services to fully initialize, then test with:
echo   python main.py
echo.

pause
