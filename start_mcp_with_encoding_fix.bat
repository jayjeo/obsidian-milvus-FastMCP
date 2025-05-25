@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo MCP SERVER STARTUP WITH ENCODING FIX
echo ============================================================
echo.
echo This script fixes the Unicode/CP949 encoding issues
echo that cause MCP server crashes on Korean Windows systems.
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Step 1: Setting up proper encoding environment...
echo --------------------------------------------------------

REM Set UTF-8 encoding for console
chcp 65001 >nul 2>&1

REM Set Python environment variables to handle encoding
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo Console encoding set to UTF-8
echo Python environment variables configured
echo.

echo Step 2: Testing Python with encoding fix...
echo --------------------------------------------------------

setlocal enabledelayedexpansion

REM Find working Python
set WORKING_PYTHON=

REM Fallback to testing common Python commands
for %%p in (python python3 py) do (
    %%p -c "print('Python encoding test OK')" >nul 2>&1
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

echo Step 3: Testing MCP server with encoding fix...
echo --------------------------------------------------------

echo Testing MCP server startup with Unicode support...
echo.

REM Create a test script to verify encoding
echo import sys > test_encoding.py
echo import os >> test_encoding.py
echo print('Python executable:', sys.executable) >> test_encoding.py
echo print('Default encoding:', sys.getdefaultencoding()) >> test_encoding.py
echo print('Console encoding:', sys.stdout.encoding) >> test_encoding.py
echo print('Filesystem encoding:', sys.getfilesystemencoding()) >> test_encoding.py
echo print('Environment PYTHONIOENCODING:', os.environ.get('PYTHONIOENCODING', 'Not set')) >> test_encoding.py
echo print('Encoding test: Hello World - 안녕하세요') >> test_encoding.py

echo Running encoding test...
"%WORKING_PYTHON%" test_encoding.py

if %errorlevel% equ 0 (
    echo Encoding test passed
) else (
    echo Encoding test failed
    echo Trying alternative encoding settings...
    
    REM Try with different encoding settings
    set PYTHONIOENCODING=utf-8:replace
    "%WORKING_PYTHON%" test_encoding.py
)

del test_encoding.py 2>nul

echo.
echo Step 4: Starting MCP server with encoding protection...
echo --------------------------------------------------------

echo Starting MCP server with Unicode/emoji protection...

REM Start MCP server with proper encoding
"%WORKING_PYTHON%" mcp_server.py

echo.
echo MCP Server process completed.
echo.

if %errorlevel% equ 0 (
    echo MCP Server completed successfully
) else (
    echo MCP Server exited with error code: %errorlevel%
    echo.
    echo The encoding fix has been applied. If you still see Unicode errors:
    echo 1. Check if your Python installation supports UTF-8
    echo 2. Consider updating to Python 3.7+
    echo 3. Try running with different encoding settings
)

echo.
pause
