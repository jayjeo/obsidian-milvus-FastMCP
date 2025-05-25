@echo off
echo ================================================================
echo Milvus Data Migration Script
echo ================================================================
echo.
echo This script will move data from G:\volumes and G:\MilvusData
echo to the project folder volumes and MilvusData directories.
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
echo Step 2: Data migration...
echo ================================================================

set /p confirm="Continue with migration? (y/N): "
if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo 2.1. Moving G:\volumes\etcd data...
echo ================================================================

if exist "G:\volumes\etcd" (
    echo Source: G:\volumes\etcd
    echo Target: %~dp0volumes\etcd
    xcopy "G:\volumes\etcd" "%~dp0volumes\etcd" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% gtr 4 (
        echo [ERROR] etcd data move failed (error code: %errorlevel%)
        goto :error
    ) else (
        echo [OK] etcd data moved successfully
    )
) else (
    echo [WARNING] G:\volumes\etcd folder does not exist.
)

echo.
echo ================================================================
echo 2.2. Moving G:\MilvusData\minio data...
echo ================================================================

if exist "G:\MilvusData\minio" (
    echo Source: G:\MilvusData\minio
    echo Target: %~dp0MilvusData\minio
    xcopy "G:\MilvusData\minio" "%~dp0MilvusData\minio" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% gtr 4 (
        echo [ERROR] minio data move failed (error code: %errorlevel%)
        goto :error
    ) else (
        echo [OK] minio data moved successfully
    )
) else (
    echo [WARNING] G:\MilvusData\mino folder does not exist.
)

echo.
echo ================================================================
echo 2.3. Moving G:\MilvusData\milvus data...
echo ================================================================

if exist "G:\MilvusData\milvus" (
    echo Source: G:\MilvusData\milvus
    echo Target: %~dp0MilvusData\milvus
    xcopy "G:\MilvusData\milvus" "%~dp0MilvusData\milvus" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% gtr 4 (
        echo [ERROR] milvus data move failed (error code: %errorlevel%)
        goto :error
    ) else (
        echo [OK] milvus data moved successfully
    )
) else (
    echo [WARNING] G:\MilvusData\milvus folder does not exist.
)

echo.
echo ================================================================
echo 2.4. Checking for additional data...
echo ================================================================

if exist "G:\volumes\minio" (
    echo Found G:\volumes\minio - moving to MilvusData\minio
    xcopy "G:\volumes\minio" "%~dp0MilvusData\minio" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% leq 4 (
        echo [OK] volumes\mino data moved successfully
    ) else (
        echo [WARNING] volumes\minio data move had issues (error code: %errorlevel%)
    )
)

if exist "G:\volumes\milvus" (
    echo Found G:\volumes\milvus - moving to MilvusData\milvus
    xcopy "G:\volumes\milvus" "%~dp0MilvusData\milvus" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% leq 4 (
        echo [OK] volumes\milvus data moved successfully
    ) else (
        echo [WARNING] volumes\milvus data move had issues (error code: %errorlevel%)
    )
)

if exist "G:\MilvusData\etcd" (
    echo Found G:\MilvusData\etcd - moving to volumes\etcd
    xcopy "G:\MilvusData\etcd" "%~dp0volumes\etcd" /E /I /H /Y
    echo DEBUG: xcopy errorlevel = %errorlevel%
    if %errorlevel% leq 4 (
        echo [OK] MilvusData\etcd data moved successfully
    ) else (
        echo [WARNING] MilvusData\etcd data move had issues (error code: %errorlevel%)
    )
)

echo.
echo ================================================================
echo Migration completed successfully!
echo ================================================================
echo.
echo Next steps:
echo 1. Verify the moved data
echo 2. Run start-milvus.bat to restart Milvus
echo 3. Test that everything works correctly
echo 4. If all is good, run cleanup-old-data.bat to delete old folders
echo.
echo Data moved to:
echo - etcd: %~dp0volumes\etcd
echo - mino: %~dp0MilvusData\minio  
echo - milvus: %~dp0MilvusData\milvus
echo.

set /p restart="Do you want to restart Milvus now? (y/N): "
if /i "%restart%"=="y" (
    echo.
    echo Starting Milvus containers...
    podman compose -f milvus-podman-compose.yml up -d
    if %errorlevel% equ 0 (
        echo [OK] Milvus containers started successfully
        echo.
        echo Please test your system to make sure everything works.
    ) else (
        echo [ERROR] Failed to start Milvus containers
    )
)

echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Migration failed.
echo Original data is preserved.
echo Please check the error and try again.
pause
exit /b 1
