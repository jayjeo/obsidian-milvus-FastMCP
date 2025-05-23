@echo off
echo ================================================================
echo           Container Name Conflict Fix
echo      충돌하는 컨테이너를 제거하고 새로 시작합니다
echo ================================================================
echo.

echo Stopping and removing conflicting containers...

echo [1/3] Removing milvus-etcd...
podman stop milvus-etcd 2>nul
podman rm milvus-etcd 2>nul

echo [2/3] Removing milvus-minio...  
podman stop milvus-minio 2>nul
podman rm milvus-minio 2>nul

echo [3/3] Removing milvus-standalone...
podman stop milvus-standalone 2>nul
podman rm milvus-standalone 2>nul

echo.
echo Cleaning up pods and networks...
podman pod stop --all 2>nul
podman pod rm --all --force 2>nul
podman network rm milvus 2>nul

echo.
echo ================================================================
echo           Container conflicts resolved!
echo ================================================================
echo.
echo Now you can run: python main.py
echo.

pause
