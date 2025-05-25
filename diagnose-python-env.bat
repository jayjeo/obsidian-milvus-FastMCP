@echo off
echo ================================================================
echo          Python Environment Diagnostic Tool
echo ================================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo ================================================================
echo Current Execution Mode
echo ================================================================

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running as ADMINISTRATOR
    echo This might cause Python environment issues
) else (
    echo [INFO] Running as NORMAL USER
)
echo.

echo ================================================================
echo Python Environment Information
echo ================================================================

echo Checking Python executable location...
where python
echo.

echo Python version:
python --version
echo.

echo Python executable path:
python -c "import sys; print('Executable:', sys.executable)"
echo.

echo Python paths:
python -c "import sys; [print('Path:', p) for p in sys.path if p]"
echo.

echo ================================================================
echo Package Installation Locations
echo ================================================================

echo Checking site-packages locations:
python -c "import site; [print('Site-packages:', p) for p in site.getsitepackages()]"
echo.

echo User site-packages:
python -c "import site; print('User site:', site.getusersitepackages())"
echo.

echo ================================================================
echo Testing Critical Imports
echo ================================================================

echo Testing markdown import:
python -c "import markdown; print('✅ markdown version:', markdown.__version__); print('   Location:', markdown.__file__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ markdown import FAILED
    echo.
    echo Checking if markdown is installed:
    python -c "import pkg_resources; print([p for p in pkg_resources.working_set if 'markdown' in p.project_name.lower()])"
)
echo.

echo Testing other critical imports:
python -c "try: import pymilvus; print('✅ pymilvus OK') except: print('❌ pymilvus FAILED')"
python -c "try: import sentence_transformers; print('✅ sentence_transformers OK') except: print('❌ sentence_transformers FAILED')"
python -c "try: import watchdog; print('✅ watchdog OK') except: print('❌ watchdog FAILED')"
python -c "try: import colorama; print('✅ colorama OK') except: print('❌ colorama FAILED')"
echo.

echo ================================================================
echo Installed Packages List
echo ================================================================

echo All installed packages:
python -m pip list | findstr -i "markdown pymilvus sentence torch"
echo.

echo ================================================================
echo Environment Variables
echo ================================================================

echo PATH (Python related):
echo %PATH% | findstr -i python
echo.

echo PYTHONPATH:
echo %PYTHONPATH%
echo.

echo PYTHONHOME:
echo %PYTHONHOME%
echo.

echo ================================================================
echo Recommendations
echo ================================================================

echo.
echo If running as administrator and packages are missing:
echo.
echo Option 1: Install packages for all users
echo   python -m pip install --upgrade markdown pymilvus sentence-transformers
echo.
echo Option 2: Run without administrator privileges
echo   (Most MCP operations don't require admin rights)
echo.
echo Option 3: Use specific Python path
echo   Check which Python has the packages installed
echo.

pause