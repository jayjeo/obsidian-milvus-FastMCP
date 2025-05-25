@echo off
echo ================================================================
echo Stop Milvus Containers
echo ================================================================
echo.
echo Stopping all Milvus containers...
podman compose -f milvus-podman-compose.yml down

if %errorlevel% equ 0 (
    echo [OK] Milvus containers stopped successfully
) else (
    echo [ERROR] Failed to stop containers (error code: %errorlevel%)
    echo This might mean no containers were running.
)

echo.
echo Checking container status...
podman ps -a --filter "name=milvus"

echo.
pause
