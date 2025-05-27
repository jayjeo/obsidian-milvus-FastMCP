@echo off
echo ================================================================
echo          Direct Conda NumPy Fix (Miniconda Environment)
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Since Miniconda is already installed, we'll use conda directly.
echo This script will:
echo 1. Check current package status
echo 2. Install NumPy using conda instead of pip
echo 3. Reinstall sentence-transformers
echo.

echo Press any key to continue...
pause

echo.
echo ================================================================
echo Step 1: Check current environment
echo ================================================================

echo Python location:
where python
echo.

echo Current NumPy status:
python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>nul || echo NumPy not installed

echo Current conda environments:
conda info --envs

echo.
echo ================================================================
echo Step 2: Remove existing packages (using conda)
echo ================================================================

echo Removing packages with conda...
conda remove -y numpy sentence-transformers --force-remove
echo.

echo Also removing with pip (in case any remain)...
python -m pip uninstall -y numpy sentence-transformers

echo.
echo ================================================================
echo Step 3: Install NumPy with conda
echo ================================================================

echo Attempting NumPy installation from default channel...
conda install -y numpy=1.26.4

if %errorlevel% neq 0 (
    echo.
    echo Default channel failed, trying conda-forge channel...
    conda install -y -c conda-forge numpy=1.26.4
    
    if %errorlevel% neq 0 (
        echo.
        echo Specific version failed, trying compatible version range...
        conda install -y -c conda-forge "numpy>=1.21.0,<2.0.0"
        
        if %errorlevel% neq 0 (
            echo ERROR: NumPy installation failed even with conda
            echo.
            echo Alternative: Create new conda environment
            echo conda create -n numpy_fix python=3.9 numpy=1.26.4
            echo conda activate numpy_fix
            pause
            exit /b 1
        )
    )
)

echo.
echo ================================================================
echo Step 4: Install sentence-transformers
echo ================================================================

echo Attempting sentence-transformers installation with conda...
conda install -y -c conda-forge sentence-transformers

if %errorlevel% neq 0 (
    echo.
    echo conda installation failed, using pip in conda environment...
    python -m pip install sentence-transformers>=2.2.2
    
    if %errorlevel% neq 0 (
        echo ERROR: sentence-transformers installation failed
        pause
        exit /b 1
    )
)

echo.
echo ================================================================
echo Step 5: Verify installation
echo ================================================================

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import numpy; print(f'NumPy location: {numpy.__file__}')"
python -c "import sentence_transformers; print('Sentence Transformers imported successfully')"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo SUCCESS! NumPy compatibility issue resolved!
    echo ================================================================
    echo.
    echo You can now run the incremental embedding option again.
    echo.
    echo Installed package information:
    conda list numpy
    conda list sentence-transformers
) else (
    echo.
    echo ERROR: Installation verification failed
    echo Please check the error messages above.
)

echo.
pause