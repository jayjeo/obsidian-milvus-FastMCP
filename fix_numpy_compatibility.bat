@echo off
echo ================================================================
echo          NumPy Compatibility Fix using Conda
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo This script will fix the NumPy compatibility issue using Conda by:
echo 1. Uninstalling current NumPy and sentence-transformers
echo 2. Installing compatible NumPy version via conda
echo 3. Installing sentence-transformers via conda/pip
echo.

echo Press any key to continue or Ctrl+C to cancel...
pause

echo.
echo ================================================================
echo Step 1: Checking conda installation
echo ================================================================

conda --version
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo.
    echo Please install Anaconda or Miniconda first:
    echo - Anaconda: https://www.anaconda.com/products/distribution
    echo - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 2: Uninstalling current packages
echo ================================================================

echo Removing with pip first...
python -m pip uninstall -y numpy sentence-transformers

echo Removing with conda...
conda remove -y numpy sentence-transformers

echo.
echo ================================================================
echo Step 3: Installing NumPy via conda
echo ================================================================

conda install -y numpy=1.26.4
if %errorlevel% neq 0 (
    echo.
    echo Trying with conda-forge channel...
    conda install -y -c conda-forge numpy=1.26.4
    if %errorlevel% neq 0 (
        echo.
        echo Trying flexible version constraint...
        conda install -y "numpy>=1.21.0,<2.0.0"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install NumPy via conda
            pause
            exit /b 1
        )
    )
)

echo.
echo ================================================================
echo Step 4: Installing sentence-transformers
echo ================================================================

echo Trying conda first...
conda install -y sentence-transformers
if %errorlevel% neq 0 (
    echo.
    echo Conda install failed, trying conda-forge...
    conda install -y -c conda-forge sentence-transformers
    if %errorlevel% neq 0 (
        echo.
        echo Conda methods failed, using pip with conda environment...
        python -m pip install --no-cache-dir sentence-transformers>=2.2.2
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install sentence-transformers
            pause
            exit /b 1
        )
    )
)

echo.
echo ================================================================
echo Step 5: Verifying installation
echo ================================================================

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers imported successfully')"
python -c "import numpy; print(f'NumPy install location: {numpy.__file__}')"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo SUCCESS: NumPy compatibility issue fixed with Conda!
    echo ================================================================
    echo.
    echo You can now run the incremental embedding option again.
    echo.
    echo Conda environment info:
    conda info --envs
) else (
    echo.
    echo ERROR: Installation verification failed
    echo Please check the error messages above
)

echo.
pause