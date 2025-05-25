@echo off
echo ================================================================
echo Milvus Data Backup Script
echo ================================================================
echo.
echo This script will backup existing G:\volumes and G:\MilvusData
echo.

set "timestamp=%random%"
set "backup_dir=%~dp0backup_%timestamp%"

echo Backup folder: %backup_dir%
echo.

echo ================================================================
echo Step 1: Stopping Milvus containers...
echo ================================================================
echo Stopping all Milvus containers...
podman compose -f milvus-podman-compose.yml down
if %errorlevel% equ 0 (
    echo [OK] Milvus containers stopped successfully
) else (
    echo [WARNING] Failed to stop containers or no containers running
)

echo.
echo ================================================================
echo Step 2: Creating backup...
echo ================================================================

set /p confirm="Start backup? (y/N): "
if /i not "%confirm%"=="y" (
    echo Backup cancelled.
    pause
    exit /b 1
)

mkdir "%backup_dir%"

echo.
echo Backup in progress...

if exist "G:\volumes" (
    echo Backing up G:\volumes...
    xcopy "G:\volumes" "%backup_dir%\volumes" /E /I /H /Y
    if %errorlevel% equ 0 (
        echo [OK] volumes backup completed
    ) else (
        echo [ERROR] volumes backup failed (error code: %errorlevel%)
    )
) else (
    echo [WARNING] G:\volumes folder does not exist.
)

echo.
if exist "G:\MilvusData" (
    echo Backing up G:\MilvusData...
    xcopy "G:\MilvusData" "%backup_dir%\MilvusData" /E /I /H /Y
    if %errorlevel% equ 0 (
        echo [OK] MilvusData backup completed
    ) else (
        echo [ERROR] MilvusData backup failed (error code: %errorlevel%)
    )
) else (
    echo [WARNING] G:\MilvusData folder does not exist.
)

echo.
echo ================================================================
echo Backup completed!
echo ================================================================
echo.
echo Backup location: %backup_dir%
echo.
echo Now you can run migrate-data.bat to move the data.
echo.
pause
