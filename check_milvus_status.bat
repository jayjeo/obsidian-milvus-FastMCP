@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ================================================================
echo              Milvus Status Checker
echo ================================================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Find Podman executable using config.py first, then PODMAN_CONFIGS
python -c "import sys; sys.path.append('.'); from config import get_podman_path; print(get_podman_path())" > temp_podman_path.txt 2>nul
if %errorlevel% equ 0 (
    set /p PODMAN_PATH=<temp_podman_path.txt
    del temp_podman_path.txt >nul 2>&1
    if exist "!PODMAN_PATH!" (
        echo Found Podman via config: !PODMAN_PATH!
        goto :check_status
    )
)

REM Clean up temp file
if exist temp_podman_path.txt del temp_podman_path.txt >nul 2>&1

echo Config detection failed, trying PODMAN_CONFIGS from config.py...

REM Try paths from PODMAN_CONFIGS in config.py
REM Try system install path from config
for /f "tokens=*" %%p in ('python -c "import os; from config import PODMAN_CONFIGS; print(PODMAN_CONFIGS['system_install']['podman_path'])" 2^>nul') do (
    set "SYSTEM_PODMAN_PATH=%%p"
)
if exist "!SYSTEM_PODMAN_PATH!" (
    set "PODMAN_PATH=!SYSTEM_PODMAN_PATH!"
    echo Found Podman (system install from config): !PODMAN_PATH!
    goto :check_status
)

REM Try user install path from config
for /f "tokens=*" %%p in ('python -c "import os; from config import PODMAN_CONFIGS; print(PODMAN_CONFIGS['user_install']['podman_path'].format(username=os.environ['USERNAME']))" 2^>nul') do (
    set "USER_PODMAN_PATH=%%p"
)
if exist "!USER_PODMAN_PATH!" (
    set "PODMAN_PATH=!USER_PODMAN_PATH!"
    echo Found Podman (user install from config): !PODMAN_PATH!
    goto :check_status
)

REM Try portable install from config (uses PATH)
for /f "tokens=*" %%p in ('python -c "from config import PODMAN_CONFIGS; print(PODMAN_CONFIGS['portable']['podman_path'])" 2^>nul') do (
    set "PORTABLE_PODMAN=%%p"
)
if "!PORTABLE_PODMAN!"=="podman" (
    for /f "tokens=*" %%q in ('where podman 2^>nul') do (
        set "PODMAN_PATH=%%q"
        echo Found Podman (portable from config): !PODMAN_PATH!
        goto :check_status
    )
)

REM If still not found, use default
if "!PODMAN_PATH!"=="" (
    echo Warning: Podman not found in any PODMAN_CONFIGS location, using default
    set "PODMAN_PATH=podman"
)

:check_status

echo Using Podman: %PODMAN_PATH%
echo.

REM Check Podman machine status
echo Podman Machine Status:
"%PODMAN_PATH%" machine list
echo.

REM Check container status
echo Container Status:
"%PODMAN_PATH%" ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

REM Check Milvus connection
echo Testing Milvus Connection:

REM Create temporary Python script
echo try: > temp_test_milvus.py
echo     from pymilvus import connections >> temp_test_milvus.py
echo     connections.connect('default', host='localhost', port='19530') >> temp_test_milvus.py
echo     print('[OK] Milvus is responding on localhost:19530') >> temp_test_milvus.py
echo     connections.disconnect('default') >> temp_test_milvus.py
echo except ImportError: >> temp_test_milvus.py
echo     print('[Warning] pymilvus not installed - cannot test connection') >> temp_test_milvus.py
echo     print('Install with: pip install pymilvus') >> temp_test_milvus.py
echo except Exception as e: >> temp_test_milvus.py
echo     print('[ERROR] Milvus connection failed:', str(e)) >> temp_test_milvus.py
echo     print('Milvus may not be running or still starting up') >> temp_test_milvus.py

python temp_test_milvus.py
del temp_test_milvus.py >nul 2>&1

echo.
echo Service URLs:
echo   - Milvus API:    http://localhost:19530
echo   - Web Interface: http://localhost:9091
echo.

REM Check log files
if exist "milvus_auto_startup.log" (
    echo Recent auto-startup log:
    powershell -command "Get-Content 'milvus_auto_startup.log' | Select-Object -Last 5"
    echo.
)

if exist "podman_startup.log" (
    echo Recent podman startup log:
    powershell -command "Get-Content 'podman_startup.log' | Select-Object -Last 3"
    echo.
)

echo ================================================================
pause
