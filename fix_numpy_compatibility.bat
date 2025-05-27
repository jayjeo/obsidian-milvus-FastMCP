@echo off
echo ================================================================
echo          Improved NumPy Compatibility Fix
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo This script will fix the NumPy compatibility issue by:
echo 1. Installing pre-compiled NumPy wheel (no compilation needed)
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
echo Step 2: Installing pre-compiled NumPy (avoiding compilation)
echo ================================================================

REM Try to install pre-compiled wheel first
python -m pip install --only-binary=all "numpy>=1.21.0,<2.0.0"
if %errorlevel% neq 0 (
    echo.
    echo Pre-compiled wheel failed, trying specific version...
    python -m pip install --only-binary=all numpy==1.26.4
    if %errorlevel% neq 0 (
        echo.
        echo Trying alternative approach with pip cache refresh...
        python -m pip install --upgrade pip
        python -m pip cache purge
        python -m pip install --only-binary=all --force-reinstall numpy==1.26.4
        if %errorlevel% neq 0 (
            echo.
            echo ERROR: All NumPy installation methods failed
            echo.
            echo SOLUTION: Please install Microsoft Visual Studio Build Tools
            echo Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
            echo Or install a conda distribution like Anaconda/Miniconda
            pause
            exit /b 1
        )
    )
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