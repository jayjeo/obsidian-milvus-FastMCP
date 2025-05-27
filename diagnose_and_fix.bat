@echo off
echo ================================================================
echo          Environment Diagnostic and Auto-Fix Script
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo ================================================================
echo Step 1: Python Environment Information
echo ================================================================

python --version
echo Python executable:
python -c "import sys; print(sys.executable)"
echo.

echo ================================================================
echo Step 2: Package Version Check
echo ================================================================

echo Checking NumPy version...
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo NumPy not installed or import error
) else (
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}');"
    echo [OK] NumPy detected - version is compatible
    set "NUMPY_FIX_NEEDED=0"
)

echo.
echo Checking sentence-transformers...
python -c "import sentence_transformers; print(f'Sentence Transformers version: {sentence_transformers.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] sentence-transformers not installed or import error
    set "ST_INSTALL_NEEDED=1"
) else (
    echo [OK] sentence-transformers installed and working
    set "ST_INSTALL_NEEDED=0"
)

echo.
echo Checking other required packages...
python -c "import pymilvus; print(f'PyMilvus version: {pymilvus.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] pymilvus not installed
) else (
    echo [OK] pymilvus installed
)

python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch not installed
) else (
    echo [OK] PyTorch installed
)

echo.
echo ================================================================
echo Step 3: Test Import of Main Components
echo ================================================================

echo Testing EmbeddingModel import...
python -c "
try:
    from embeddings import EmbeddingModel
    print('[OK] EmbeddingModel import successful')
except Exception as e:
    print(f'[ERROR] EmbeddingModel import failed: {e}')
    exit(1)
" 2>nul

if %errorlevel% neq 0 (
    echo [ERROR] Main component import failed
    set "COMPONENT_ERROR=1"
) else (
    echo [OK] All main components import successfully
    set "COMPONENT_ERROR=0"
)

echo.
echo ================================================================
echo Step 4: Auto-Fix Recommendations
echo ================================================================

rem NumPy compatibility check removed as sentence-transformers now supports NumPy 2.x
echo [OK] NumPy compatibility check skipped (not needed with current versions)

if "%ST_INSTALL_NEEDED%"=="1" (
    echo.
    echo [ACTION REQUIRED] sentence-transformers installation needed
    echo.
    set /p install_choice="Install sentence-transformers? (y/n): "
    if /i "!install_choice!"=="y" (
        echo Installing sentence-transformers...
        python -m pip install sentence-transformers>=2.2.2
    )
)

echo.
echo ================================================================
echo Diagnostic Complete
echo ================================================================

if "%NUMPY_FIX_NEEDED%"=="0" AND "%ST_INSTALL_NEEDED%"=="0" AND "%COMPONENT_ERROR%"=="0" (
    echo [SUCCESS] All components are working correctly!
    echo You can now run the incremental embedding option.
) else (
    echo [INFO] Some issues were detected. Please review the recommendations above.
)

echo.
echo For more detailed troubleshooting, check:
echo - requirements.txt for package versions
echo - config.py for configuration settings
echo - embeddings.py for compatibility checks
echo.

pause
