@echo off
echo ================================================================
echo                Complete Milvus System Reset
echo              This will fix all container conflicts
echo ================================================================
echo.
echo [WARNING] This script will affect:
echo   - ALL Podman containers system-wide (not just Milvus)
echo   - Local directories: %CD%\MilvusData and %CD%\volumes
echo   - All Podman volumes, networks, and pods
echo.
echo Current directory: %CD%
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/5] Killing all running containers...
podman kill --all 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/5] Removing all containers forcefully...
podman rm --all --force 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [3/5] Removing all pods...
podman pod rm --all --force 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [4/5] Cleaning up volumes and networks...
podman volume rm --all --force 2>nul
podman network prune --force 2>nul
podman system prune --all --force --volumes 2>nul
timeout /t 3 /nobreak >nul

echo.
echo [5/5] Removing local data directories...
if exist "MilvusData" (
    echo Removing MilvusData from: %CD%\MilvusData
    rmdir /s /q "MilvusData" 2>nul
)

if exist "volumes" (
    echo Removing volumes from: %CD%\volumes
    rmdir /s /q "volumes" 2>nul
)

echo.
echo ================================================================
echo                    RESET COMPLETE!
echo ================================================================
echo.
echo The system is now completely clean. You can now run:
echo   1. python main.py
echo   2. Choose option 2 (Full Embedding)
echo   3. Select Yes for "erase all"
echo.

pause
