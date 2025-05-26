@echo off
echo ================================================================
echo         Finding Podman and Compose Paths for config.json
echo ================================================================
echo.

echo Searching for Podman...
where podman >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Podman not found in PATH
    echo Please install Podman first: winget install RedHat.Podman
    goto :end
)

for /f "tokens=*" %%i in ('where podman') do set PODMAN_PATH=%%i
echo ✅ Podman found: %PODMAN_PATH%

echo.
echo Searching for Compose providers...

REM Check for podman-compose
where podman-compose >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('where podman-compose') do set PODMAN_COMPOSE_PATH=%%i
    echo ✅ podman-compose found: !PODMAN_COMPOSE_PATH!
) else (
    echo ⚠️  podman-compose not found
    echo    You can install it with: pip install podman-compose
)

REM Check for podman-compose
where podman-compose >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('where podman-compose') do set PODMAN_COMPOSE_PATH=%%i
    echo ✅ podman-compose found: !PODMAN_COMPOSE_PATH!
) else (
    echo ⚠️  podman-compose not found
)

echo.
echo ================================================================
echo                    Config.json Examples
echo ================================================================

echo.
echo **Option 1: Using podman-compose**
echo {
echo   "podman_path": "%PODMAN_PATH:\=\\%",
if defined PODMAN_COMPOSE_PATH (
    echo   "podman_compose_path": "!PODMAN_COMPOSE_PATH:\=\\!"
)
echo }

echo.
echo **Option 2: Using podman-compose with Podman**
echo {
echo   "podman_path": "%PODMAN_PATH:\=\\%",
if defined PODMAN_COMPOSE_PATH (
    echo   "compose_path": "!PODMAN_COMPOSE_PATH:\=\\!"
)
echo }

echo.
echo **Option 3: Using Podman built-in compose**
echo {
echo   "podman_path": "%PODMAN_PATH:\=\\%",
echo   "compose_command": "podman compose"
echo }

echo.
echo ================================================================
echo Copy the appropriate configuration to your config.json file
echo ================================================================

:end
echo.
pause