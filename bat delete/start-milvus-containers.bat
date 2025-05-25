@echo off
echo ================================================================
echo Start Milvus Containers
echo ================================================================
echo.
echo Starting all Milvus containers...
podman compose -f milvus-podman-compose.yml up -d

if %errorlevel% equ 0 (
    echo [OK] Milvus containers started successfully
) else (
    echo [ERROR] Failed to start containers (error code: %errorlevel%)
)

echo.
echo Checking container status...
podman ps --filter "name=milvus"

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak > nul

echo.
echo Container logs (last 10 lines):
echo --------------------------------
podman logs --tail 10 milvus-standalone 2>nul
echo.
pause
