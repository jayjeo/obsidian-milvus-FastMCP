@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo TESTING EMBEDDINGS FIX
echo ============================================================
echo.
echo This script tests if the hanging issue in embeddings.py is fixed
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo Finding Python executable...
setlocal enabledelayedexpansion

REM Find working Python
set WORKING_PYTHON=

for %%p in (python python3 py) do (
    %%p -c "print('Python found')" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('%%p -c "import sys; print(sys.executable)"') do (
            set WORKING_PYTHON=%%i
        )
        echo Found working Python: %%p
        goto found_python
    )
)

echo No working Python found
pause
exit /b 1

:found_python
echo Working Python: %WORKING_PYTHON%
echo.

echo Running embeddings test...
echo This will test if the hanging issue is fixed
echo.

"%WORKING_PYTHON%" test_embeddings_fix.py

echo.
echo Test completed.
pause
