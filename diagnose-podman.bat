@echo off
echo ================================================================
echo               Podman and Milvus Diagnostic Tool
echo ================================================================
echo.

echo [INFO] Checking Podman installation...
where podman >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Podman is not installed or not in PATH
    goto :end
)

echo [OK] Podman found:
podman --version
echo.

echo [INFO] Checking Podman machine status...
podman machine list
echo.

echo [INFO] Checking all containers...
podman ps --all
echo.

echo [INFO] Checking all pods...
podman pod list
echo.

echo [INFO] Checking volumes...
podman volume list
echo.

echo [INFO] Checking networks...
podman network list
echo.

echo [INFO] Checking Milvus-specific containers...
podman ps --all --filter name=milvus
echo.

echo [INFO] Checking system disk usage...
podman system df
echo.

echo [INFO] Checking for port conflicts...
netstat -an | findstr :19530
netstat -an | findstr :9091
echo.

echo [INFO] Testing Milvus connection...
powershell -Command "try { $null = Test-NetConnection -ComputerName localhost -Port 19530 -InformationLevel Quiet; if ($?) { Write-Host '[OK] Port 19530 is reachable' } else { Write-Host '[WARNING] Port 19530 is not reachable' } } catch { Write-Host '[ERROR] Failed to test port 19530' }"

echo.
echo ================================================================
echo                    Diagnostic Complete
echo ================================================================

:end
pause
