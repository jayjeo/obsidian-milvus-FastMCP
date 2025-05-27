@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo          Clean NumPy Installation with Conda
echo ================================================================
echo.

REM Change to script directory
cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Check if conda is available
conda --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    echo.
    goto :error_exit
)

echo Conda version check: OK
echo.

echo This script will:
echo 1. Remove any existing numpy/sentence-transformers
echo 2. Install numpy using conda (no compilation needed)
echo 3. Install sentence-transformers
echo 4. Verify installation
echo.

echo Press any key to start, or close window to cancel...
pause >nul

echo.
echo ================================================================
echo Step 1: Clean removal of existing packages
echo ================================================================

echo Removing with conda...
conda remove -y numpy sentence-transformers 2>nul
echo Removing with pip (backup)...
python -m pip uninstall -y numpy sentence-transformers 2>nul

echo Cleanup completed.
echo.

echo ================================================================
echo Step 2: Install NumPy with conda
echo ================================================================

echo Installing numpy=1.26.4 from conda...
conda install -y numpy=1.26.4

if !errorlevel! neq 0 (
    echo Default channel failed, trying conda-forge...
    conda install -y -c conda-forge numpy=1.26.4
    
    if !errorlevel! neq 0 (
        echo Specific version failed, trying compatible range...
        conda install -y -c conda-forge "numpy>=1.21.0,<2.0.0"
        
        if !errorlevel! neq 0 (
            echo ERROR: All conda NumPy installation attempts failed
            goto :error_exit
        )
    )
)

echo NumPy installation: SUCCESS
echo.

echo ================================================================
echo Step 3: Install sentence-transformers
echo ================================================================

echo Installing sentence-transformers with conda...
conda install -y -c conda-forge sentence-transformers

if !errorlevel! neq 0 (
    echo Conda failed, using pip in conda environment...
    python -m pip install sentence-transformers>=2.2.2
    
    if !errorlevel! neq 0 (
        echo ERROR: sentence-transformers installation failed
        goto :error_exit
    )
)

echo sentence-transformers installation: SUCCESS
echo.

echo ================================================================
echo Step 4: Verification
echo ================================================================

echo Testing NumPy import...
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>nul
if !errorlevel! neq 0 (
    echo ERROR: NumPy verification failed
    goto :error_exit
)

echo Testing sentence-transformers import...
python -c "import sentence_transformers; print('sentence-transformers: OK')" 2>nul
if !errorlevel! neq 0 (
    echo ERROR: sentence-transformers verification failed
    goto :error_exit
)

echo.
echo ================================================================
echo SUCCESS: All packages installed and working!
echo ================================================================
echo.

echo Package information:
conda list numpy sentence-transformers

echo.
echo You can now run your incremental embedding script.
echo.
goto :success_exit

:error_exit
echo.
echo ================================================================
echo Installation failed. Manual alternative:
echo ================================================================
echo.
echo Try these commands manually:
echo 1. conda create -n numpy_env python=3.9
echo 2. conda activate numpy_env  
echo 3. conda install numpy=1.26.4
echo 4. pip install sentence-transformers
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:success_exit
echo Press any key to exit...
pause >nul
exit /b 0