@echo off
echo ================================================================
echo      Quick Fix for Missing Modules
echo ================================================================
echo.

cd /d "%~dp0"

echo Testing current Python and packages...
echo.

echo Python location:
python -c "import sys; print(sys.executable)"
echo.

echo Python path:
python -c "import sys; [print(p) for p in sys.path]"
echo.

echo Testing package imports:
echo.

python -c "
try:
    import pymilvus
    print('✓ pymilvus: OK')
    print(f'  Version: {pymilvus.__version__}')
    print(f'  Location: {pymilvus.__file__}')
except ImportError as e:
    print(f'✗ pymilvus: FAILED - {e}')

try:
    import mcp
    print('✓ mcp: OK')
except ImportError as e:
    print(f'✗ mcp: FAILED - {e}')

try:
    import fastmcp
    print('✓ fastmcp: OK')
except ImportError as e:
    print(f'✗ fastmcp: FAILED - {e}')

try:
    import torch
    print('✓ torch: OK')
    print(f'  Version: {torch.__version__}')
except ImportError as e:
    print(f'✗ torch: FAILED - {e}')
"

echo.
echo ================================================================
echo Trying alternative installation methods...
echo ================================================================
echo.

echo Method 1: Direct pip install in user space
python -m pip install --user pymilvus mcp fastmcp sentence-transformers
echo.

echo Method 2: Force reinstall
python -m pip install --user --force-reinstall pymilvus
echo.

echo Testing again after user installation:
python -c "
try:
    import pymilvus
    print('✓ pymilvus: NOW WORKING')
    print(f'  Version: {pymilvus.__version__}')
except ImportError as e:
    print(f'✗ pymilvus: STILL FAILED - {e}')
"

echo.
echo ================================================================
echo If still not working, try running as administrator
echo ================================================================
pause
