@echo off
echo ================================================================
echo          NumPy Compatibility Fix for Sentence Transformers
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo This script will fix the NumPy compatibility issue by:
echo 1. Downgrading NumPy to version 1.26.4 (last stable 1.x version)
echo 2. Reinstalling sentence-transformers with the compatible NumPy
echo.

echo Press any key to continue or Ctrl+C to cancel...
pause

echo.
echo ================================================================
echo Step 1: Uninstalling current NumPy and sentence-transformers
echo ================================================================

python -m pip uninstall -y numpy sentence-transformers
if %errorlevel% neq 0 (
    echo Warning: Uninstall may have had issues, continuing...
)

echo.
echo ================================================================
echo Step 2: Installing compatible NumPy version
echo ================================================================

python -m pip install "numpy<2,>=1.21.0"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install compatible NumPy
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 3: Reinstalling sentence-transformers
echo ================================================================

python -m pip install --no-cache-dir sentence-transformers>=2.2.2
if %errorlevel% neq 0 (
    echo ERROR: Failed to install sentence-transformers
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 4: Verifying installation
echo ================================================================

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers imported successfully')"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo SUCCESS: NumPy compatibility issue fixed!
    echo ================================================================
    echo.
    echo You can now run the incremental embedding option again.
) else (
    echo.
    echo ERROR: Installation verification failed
    echo Please check the error messages above
)

echo.
pause
