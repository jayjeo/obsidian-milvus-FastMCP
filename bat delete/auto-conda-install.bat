@echo off
title Conda Auto Install

echo ================================================================
echo Conda Auto Install
echo ================================================================
echo.

echo Checking conda...
conda --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Conda found!
    goto install_packages
)

echo Conda not found. Installing Miniconda...
echo.

echo Downloading installer...
powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%TEMP%\miniconda.exe'"

echo Installing Miniconda...
%TEMP%\miniconda.exe /InstallationType=JustMe /AddToPath=1 /S /D=C:\Miniconda3

echo Waiting for installation...
timeout /t 10 /nobreak >nul

echo Updating PATH...
set PATH=C:\Miniconda3\Scripts;C:\Miniconda3;%PATH%

echo Detecting conda after installation...
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Miniconda3\Scripts\conda.exe" (
        set PATH=C:\Miniconda3\Scripts;%PATH%
    )
)

:install_packages
echo.
echo ================================================================
echo Installing packages...
echo ================================================================
echo Current directory: %CD%
echo.

if exist requirements.txt (
    echo Found requirements.txt file
    echo Installing basic conda packages first...
    conda install -c conda-forge -y python pip
    echo.
    echo Installing from requirements.txt...
    conda run -n base pip install -r requirements.txt
    echo.
) else (
    echo requirements.txt not found, installing individual packages...
    echo Step 1: Basic packages
    conda install -c conda-forge -y python pip numpy pandas requests pyyaml tqdm colorama
    echo Step 2: AI packages  
    conda install -c conda-forge -y pytorch sentence-transformers
    echo Step 3: Specialized packages
    conda run -n base pip install pymilvus mcp fastmcp
    echo Step 4: Additional packages
    conda run -n base pip install PyPDF2 markdown beautifulsoup4 python-dotenv watchdog psutil
    echo.
)

echo ================================================================
echo Testing installation...
echo ================================================================
conda run -n base python -c "
try:
    import pymilvus
    print('✓ pymilvus: SUCCESS')
except ImportError as e:
    print('✗ pymilvus: FAILED -', str(e))

try:
    import mcp
    print('✓ mcp: SUCCESS')
except ImportError as e:
    print('✗ mcp: FAILED -', str(e))

try:
    import fastmcp
    print('✓ fastmcp: SUCCESS')
except ImportError as e:
    print('✗ fastmcp: FAILED -', str(e))

try:
    import torch
    print('✓ torch: SUCCESS')
except ImportError as e:
    print('✗ torch: FAILED -', str(e))
"

echo.
echo ================================================================
echo Installation completed!
echo ================================================================
pause
