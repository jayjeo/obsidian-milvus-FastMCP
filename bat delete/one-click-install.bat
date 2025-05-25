@echo off
setlocal enabledelayedexpansion
title One-Click Package Installation

echo ================================================================
echo      One-Click Package Installation System
echo ================================================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python first
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

if not exist "config.py" (
    echo ERROR: config.py not found
    echo Please ensure config.py is in the same directory
    pause
    exit /b 1
)

echo Found requirements.txt and config.py
echo.

echo Installation Menu:
echo 1. Auto-detect and install (recommended)
echo 2. Quick pip-only install
echo 3. Exit
echo.

set /p choice="Select option (1-3): "

if "%choice%"=="1" goto auto_install
if "%choice%"=="2" goto pip_install
if "%choice%"=="3" goto exit_now

:auto_install
echo.
echo Reading configuration and detecting conda/mamba...

python detect_config.py

if exist "temp_config.bat" (
    call temp_config.bat
    del temp_config.bat
) else (
    echo Failed to read configuration
    set CONDA_FOUND=0
    set MAMBA_FOUND=0
)

echo.
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip not found in PATH, trying python -m pip...
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo pip not found, attempting to install pip...
        python -m ensurepip --upgrade >nul 2>&1
        if %errorlevel% equ 0 (
            echo Pip installed successfully via ensurepip
            python -m pip --version
            set PIP_CMD=python -m pip
        ) else (
            if %CONDA_FOUND%==1 (
                echo Installing pip via conda...
                "!CONDA_PATH!" install pip -y
                python -m pip --version >nul 2>&1
                if %errorlevel% equ 0 (
                    echo Pip successfully installed via conda:
                    python -m pip --version
                    set PIP_CMD=python -m pip
                ) else (
                    echo Will use conda/mamba for all packages
                    set PIP_CMD=
                )
            ) else (
                echo ERROR: pip not found
                echo Please install pip or ensure Python installation includes pip
                echo You can try: python -m ensurepip --upgrade
                pause
                exit /b 1
            )
        )
    ) else (
        echo Pip found via python -m pip:
        python -m pip --version
        set PIP_CMD=python -m pip
    )
) else (
    echo Pip found:
    pip --version
    set PIP_CMD=pip
)
echo.

REM Force mamba installation and use
if %CONDA_FOUND%==1 (
    echo Checking for mamba installation...
    
    REM Check if mamba.exe exists in conda Scripts folder
    for %%F in ("!CONDA_PATH!") do set "CONDA_SCRIPTS_DIR=%%~dpF"
    set "MAMBA_EXE=!CONDA_SCRIPTS_DIR!mamba.exe"
    
    if exist "!MAMBA_EXE!" (
        echo Mamba executable found: !MAMBA_EXE!
        "!MAMBA_EXE!" --version >nul 2>&1
        if %errorlevel% equ 0 (
            echo Mamba is working
            set MAMBA_FOUND=1
            set MAMBA_EXE_PATH=!MAMBA_EXE!
        ) else (
            echo Mamba executable exists but not working
            set MAMBA_FOUND=0
        )
    ) else (
        echo Mamba executable not found, installing...
        "!CONDA_PATH!" install mamba -n base -c conda-forge -y
        if %errorlevel% equ 0 (
            echo Mamba installation completed
            
            REM Re-check for mamba in multiple locations after installation
            set "MAMBA_FOUND_AFTER=0"
            
            REM Check original location
            if exist "!MAMBA_EXE!" (
                "!MAMBA_EXE!" --version >nul 2>&1
                if %errorlevel% equ 0 (
                    echo Mamba found at original location: !MAMBA_EXE!
                    set MAMBA_FOUND=1
                    set MAMBA_EXE_PATH=!MAMBA_EXE!
                    set "MAMBA_FOUND_AFTER=1"
                )
            )
            
            REM Check alternative locations if not found
            if "!MAMBA_FOUND_AFTER!"=="0" (
                echo Checking alternative mamba locations...
                
                REM Try base environment Scripts folder
                for %%F in ("!CONDA_PATH!") do set "CONDA_BASE=%%~dpF"
                set "ALT_MAMBA1=!CONDA_BASE!..\Scripts\mamba.exe"
                if exist "!ALT_MAMBA1!" (
                    "!ALT_MAMBA1!" --version >nul 2>&1
                    if %errorlevel% equ 0 (
                        echo Mamba found at: !ALT_MAMBA1!
                        set MAMBA_FOUND=1
                        set MAMBA_EXE_PATH=!ALT_MAMBA1!
                        set "MAMBA_FOUND_AFTER=1"
                    )
                )
                
                REM Try conda run method if still not found
                if "!MAMBA_FOUND_AFTER!"=="0" (
                    "!CONDA_PATH!" run -n base mamba --version >nul 2>&1
                    if %errorlevel% equ 0 (
                        echo Mamba works through conda run method
                        set MAMBA_FOUND=1
                        set MAMBA_EXE_PATH=CONDA_RUN_MAMBA
                        set "MAMBA_FOUND_AFTER=1"
                    )
                )
            )
            
            if "!MAMBA_FOUND_AFTER!"=="0" (
                echo Mamba executable still not found after installation
                set MAMBA_FOUND=0
            )
        ) else (
            echo Mamba installation failed
            set MAMBA_FOUND=0
        )
    )
)

echo Installation Strategy:
if %MAMBA_FOUND%==1 (
    echo - Using Mamba for common packages (fastest)
    echo - Using Pip for specialized packages
    if "!MAMBA_EXE_PATH!"=="CONDA_RUN_MAMBA" (
        echo - Mamba: conda run method
    ) else (
        echo - Mamba: !MAMBA_EXE_PATH!
    )
) else (
    if %CONDA_FOUND%==1 (
        echo - Using Conda for common packages
        echo - Using Pip for specialized packages
        echo - Conda: !CONDA_PATH!
    ) else (
        echo - Using Pip only
    )
)
echo.

echo Packages to install:
type requirements.txt
echo.

set /p confirm="Start installation? (y/n): "
if /i not "!confirm!"=="y" (
    echo Installation cancelled
    pause
    exit /b 0
)

goto start_install

:pip_install
echo.
echo Quick pip-only installation
echo.

pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip not found in PATH, trying python -m pip...
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo pip not found, attempting to install pip...
        python -m ensurepip --upgrade >nul 2>&1
        if %errorlevel% equ 0 (
            echo Pip installed successfully via ensurepip
            python -m pip --version
            set PIP_CMD=python -m pip
        ) else (
            echo ERROR: pip not found
            echo Please install pip or ensure Python installation includes pip
            echo You can try: python -m ensurepip --upgrade
            pause
            exit /b 1
        )
    ) else (
        echo Pip found via python -m pip:
        python -m pip --version
        set PIP_CMD=python -m pip
    )
) else (
    echo Pip found:
    pip --version
    set PIP_CMD=pip
)
echo.

set /p confirm="Start pip-only installation? (y/n): "
if /i not "!confirm!"=="y" (
    echo Installation cancelled
    pause
    exit /b 0
)

set MAMBA_FOUND=0
set CONDA_FOUND=0
goto start_install

:start_install
echo.
echo Starting Installation...
echo.

if %MAMBA_FOUND%==1 (
    echo Installing common packages with Mamba...
    echo Using: !MAMBA_EXE_PATH!
    echo.
    REM Check if using conda run method and handle accordingly
    if "!MAMBA_EXE_PATH!"=="CONDA_RUN_MAMBA" (
        REM Use conda run method
        "!CONDA_PATH!" run -n base mamba install numpy pandas requests pyyaml tqdm colorama -c conda-forge -y
    ) else (
        REM Use direct mamba executable
        "!MAMBA_EXE_PATH!" install numpy pandas requests pyyaml tqdm colorama -c conda-forge -y
    )
    echo Mamba packages installation completed
    echo.
) else (
    if %CONDA_FOUND%==1 (
        echo Installing common packages with Conda...
        echo Using: !CONDA_PATH!
        echo.
        "!CONDA_PATH!" install numpy pandas requests pyyaml tqdm colorama -c conda-forge -y
        echo Conda packages installation completed
        echo.
    ) else (
        echo No conda/mamba available, using pip for all packages
        echo.
    )
)

echo Installing specialized packages with Pip...
echo.

if defined PIP_CMD (
    echo Upgrading pip...
    %PIP_CMD% install --upgrade pip
) else (
    echo Skipping pip upgrade - using conda only
)

if defined PIP_CMD (
    echo Installing PyTorch...
    %PIP_CMD% install torch

    echo Installing core packages...
    %PIP_CMD% install pymilvus mcp fastmcp sentence-transformers

    echo Installing remaining packages...
    %PIP_CMD% install -r requirements.txt --upgrade
) else (
    echo Installing all packages with conda...
    "!CONDA_PATH!" install -c conda-forge pytorch pymilvus -y
    echo Note: Some specialized packages may not be available via conda
)

echo.
echo Installation complete!
echo.

echo Testing installation...
python -c "import pymilvus, mcp, fastmcp; print('Installation verified successfully!')"
if %errorlevel% equ 0 (
    echo All packages imported successfully!
) else (
    echo Warning: Some packages may not have installed correctly
)

echo.
echo Installation method used:
if %MAMBA_FOUND%==1 (
    echo - Mamba + Pip (optimal speed)
    if "!MAMBA_EXE_PATH!"=="CONDA_RUN_MAMBA" (
        echo - Mamba: conda run method
    ) else (
        echo - Mamba: !MAMBA_EXE_PATH!
    )
) else (
    if %CONDA_FOUND%==1 (
        echo - Conda + Pip (good speed)
        echo - Conda: !CONDA_PATH!
    ) else (
        echo - Pip only (standard speed)
    )
)

echo.
echo Next steps:
echo 1. Test: python -c "import pymilvus, mcp, fastmcp"
echo 2. Run: python main.py
echo.
goto end_script

:exit_now
echo Installation cancelled
goto end_script

:end_script
pause
