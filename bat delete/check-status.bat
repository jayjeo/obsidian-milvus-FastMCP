@echo off
title Check Status

echo Current directory: %CD%
echo.

echo Checking conda...
conda --version
echo.

echo Checking requirements.txt...
if exist requirements.txt (
    echo requirements.txt found
    echo Contents:
    type requirements.txt
) else (
    echo requirements.txt NOT found
)
echo.

echo Checking Python packages...
conda run -n base python -c "
try:
    import pymilvus
    print('✓ pymilvus: OK')
except ImportError:
    print('✗ pymilvus: NOT INSTALLED')

try:
    import mcp
    print('✓ mcp: OK')
except ImportError:
    print('✗ mcp: NOT INSTALLED')

try:
    import fastmcp
    print('✓ fastmcp: OK')
except ImportError:
    print('✗ fastmcp: NOT INSTALLED')
"

echo.
echo Status check complete!
pause
