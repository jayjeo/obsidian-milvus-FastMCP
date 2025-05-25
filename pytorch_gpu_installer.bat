@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo              PyTorch GPU Version Checker and Installer
echo ================================================================
echo.
echo This script will:
echo 1. Check if PyTorch supports CUDA GPU
echo 2. Detect your CUDA version
echo 3. Reinstall GPU version if needed
echo.

REM Get current project directory
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

echo Project directory: %PROJECT_DIR%
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    echo Please install Python or add it to PATH
    pause
    exit /b 1
)

echo [SUCCESS] Python is available
for /f "tokens=*" %%v in ('python --version 2^>nul') do (
    echo Python version: %%v
)
echo.

REM Step 1: Check current PyTorch installation
echo ================================================================
echo Step 1: Checking current PyTorch installation
echo ================================================================

echo Checking if PyTorch is installed...
python -c "import torch; print('PyTorch version:', torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] PyTorch not found, will install GPU version
    goto :install_gpu_version
)

echo [SUCCESS] PyTorch is installed
python -c "import torch; print('PyTorch version:', torch.__version__)"

REM Check CUDA availability
echo.
echo Checking CUDA support in current PyTorch installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" > temp_cuda_check.txt 2>&1

findstr /i "true" temp_cuda_check.txt >nul
if %errorlevel% == 0 (
    echo [SUCCESS] PyTorch already supports CUDA GPU!
    echo.
    echo Current GPU information:
    python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available'); print('GPU count:', torch.cuda.device_count()); [print('GPU name:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
    
    del temp_cuda_check.txt >nul 2>&1
    echo.
    echo PyTorch GPU support is already working correctly!
    pause
    exit /b 0
) else (
    echo [WARNING] PyTorch found but CUDA support is NOT available
    echo Current PyTorch is CPU version only
    echo.
    python -c "import torch; print('Current PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
)

del temp_cuda_check.txt >nul 2>&1

REM Step 2: Detect CUDA version
echo.
echo ================================================================
echo Step 2: Detecting CUDA version on system
echo ================================================================

echo Checking for NVIDIA GPU and CUDA installation...

REM Check nvidia-smi
nvidia-smi --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found
    echo Checking for NVIDIA GPU in other ways...
    
    REM Try alternative GPU detection
    wmic path win32_VideoController get name | findstr /i nvidia >nul
    if %errorlevel% neq 0 (
        echo [ERROR] No NVIDIA GPU detected on this system
        echo This system may not support CUDA acceleration
        echo.
        set /p "continue_choice=Continue with CPU version installation? (y/n): "
        if /i "!continue_choice!" neq "y" (
            echo Installation cancelled
            pause
            exit /b 1
        )
        echo Installing CPU version...
        goto :install_cpu_version
    ) else (
        echo [INFO] NVIDIA GPU detected but CUDA tools not found
        echo Will attempt to install latest CUDA-compatible PyTorch
        set "CUDA_VERSION=cu121"
        goto :confirm_installation
    )
) else (
    echo [SUCCESS] NVIDIA GPU and CUDA tools detected
    
    REM Get CUDA version from nvidia-smi
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader^,nounits 2^>nul') do (
        echo NVIDIA Driver version: %%i
    )
    
    REM Try to get CUDA runtime version
    nvcc --version >nul 2>&1
    if %errorlevel% == 0 (
        echo CUDA Compiler detected:
        nvcc --version | findstr "release"
        
        REM Parse CUDA version for PyTorch compatibility
        for /f "tokens=*" %%i in ('nvcc --version ^| findstr "release" 2^>nul') do (
            echo CUDA info: %%i
            echo %%i | findstr "11.8" >nul && set "CUDA_VERSION=cu118"
            echo %%i | findstr "12.1" >nul && set "CUDA_VERSION=cu121"
            echo %%i | findstr "12.4" >nul && set "CUDA_VERSION=cu124"
        )
    ) else (
        echo [INFO] CUDA Compiler not found, using latest compatible version
        set "CUDA_VERSION=cu121"
    )
)

REM Default to cu121 if version not detected
if not defined CUDA_VERSION (
    echo [INFO] Could not determine CUDA version, using CUDA 12.1 compatible version
    set "CUDA_VERSION=cu121"
)

:confirm_installation
echo.
echo ================================================================
echo Step 3: Installation Plan
echo ================================================================

echo Current situation:
if defined PYTORCH_INSTALLED (
    echo - PyTorch is installed but CPU version only
) else (
    echo - PyTorch not installed
)
echo - Detected CUDA compatibility: %CUDA_VERSION%
echo.

REM Map CUDA version to PyTorch index URL
if "%CUDA_VERSION%"=="cu118" (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118"
    echo Will install: PyTorch with CUDA 11.8 support
) else if "%CUDA_VERSION%"=="cu121" (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"
    echo Will install: PyTorch with CUDA 12.1 support
) else if "%CUDA_VERSION%"=="cu124" (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
    echo Will install: PyTorch with CUDA 12.4 support
) else (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"
    echo Will install: PyTorch with CUDA 12.1 support (default)
)

echo PyTorch index URL: %TORCH_INDEX_URL%
echo.

set /p "install_choice=Proceed with GPU PyTorch installation? (y/n): "
if /i "%install_choice%" neq "y" (
    echo Installation cancelled by user
    pause
    exit /b 0
)

:install_gpu_version
echo.
echo ================================================================
echo Step 4: Installing PyTorch GPU Version
echo ================================================================

echo Uninstalling existing PyTorch packages...
pip uninstall torch torchvision torchaudio -y
if %errorlevel% neq 0 (
    echo [WARNING] Error during uninstall, continuing...
)

echo.
echo Installing PyTorch GPU version...
echo Command: pip install torch torchvision torchaudio --index-url %TORCH_INDEX_URL%
echo.

pip install torch torchvision torchaudio --index-url %TORCH_INDEX_URL%
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch GPU installation failed
    echo.
    echo Possible solutions:
    echo 1. Check internet connection
    echo 2. Try different CUDA version
    echo 3. Install CPU version as fallback
    echo.
    set /p "fallback_choice=Install CPU version as fallback? (y/n): "
    if /i "!fallback_choice!"=="y" goto :install_cpu_version
    pause
    exit /b 1
)

echo [SUCCESS] PyTorch GPU installation completed!
goto :verify_installation

:install_cpu_version
echo.
echo ================================================================
echo Installing PyTorch CPU Version
echo ================================================================

echo Installing CPU-only PyTorch...
pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch CPU installation failed
    pause
    exit /b 1
)

echo [SUCCESS] PyTorch CPU installation completed!

:verify_installation
echo.
echo ================================================================
echo Step 5: Verifying Installation
echo ================================================================

echo Testing PyTorch installation...
python -c "import torch; print('='*50); print('PyTorch Installation Verification'); print('='*50); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'CPU only'); print('Number of GPUs:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]; print('='*50)"

if %errorlevel% neq 0 (
    echo [ERROR] PyTorch verification failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Installation Summary
echo ================================================================

REM Create verification script for future use
set "VERIFY_SCRIPT=%PROJECT_DIR%\verify_pytorch_gpu.py"
echo Creating verification script: %VERIFY_SCRIPT%

echo # PyTorch GPU Verification Script> "%VERIFY_SCRIPT%"
echo # Generated automatically by pytorch_gpu_installer.bat>> "%VERIFY_SCRIPT%"
echo.>> "%VERIFY_SCRIPT%"
echo import torch>> "%VERIFY_SCRIPT%"
echo import sys>> "%VERIFY_SCRIPT%"
echo.>> "%VERIFY_SCRIPT%"
echo def check_pytorch_gpu():>> "%VERIFY_SCRIPT%"
echo     print("="*60)>> "%VERIFY_SCRIPT%"
echo     print("PyTorch GPU Support Verification")>> "%VERIFY_SCRIPT%"
echo     print("="*60)>> "%VERIFY_SCRIPT%"
echo     print(f"PyTorch version: {torch.__version__}")>> "%VERIFY_SCRIPT%"
echo     print(f"Python version: {sys.version}")>> "%VERIFY_SCRIPT%"
echo     print(f"CUDA available: {torch.cuda.is_available()}")>> "%VERIFY_SCRIPT%"
echo     if torch.cuda.is_available():>> "%VERIFY_SCRIPT%"
echo         print(f"CUDA version: {torch.version.cuda}")>> "%VERIFY_SCRIPT%"
echo         print(f"Number of GPUs: {torch.cuda.device_count()}")>> "%VERIFY_SCRIPT%"
echo         for i in range(torch.cuda.device_count()):>> "%VERIFY_SCRIPT%"
echo             print(f"GPU {i}: {torch.cuda.get_device_name(i)}")>> "%VERIFY_SCRIPT%"
echo             print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")>> "%VERIFY_SCRIPT%"
echo         >> "%VERIFY_SCRIPT%"
echo         # Test GPU computation>> "%VERIFY_SCRIPT%"
echo         try:>> "%VERIFY_SCRIPT%"
echo             x = torch.randn(1000, 1000).cuda()>> "%VERIFY_SCRIPT%"
echo             y = torch.matmul(x, x)>> "%VERIFY_SCRIPT%"
echo             print("GPU computation test: PASSED")>> "%VERIFY_SCRIPT%"
echo         except Exception as e:>> "%VERIFY_SCRIPT%"
echo             print(f"GPU computation test: FAILED - {e}")>> "%VERIFY_SCRIPT%"
echo     else:>> "%VERIFY_SCRIPT%"
echo         print("CUDA not available - using CPU only")>> "%VERIFY_SCRIPT%"
echo     print("="*60)>> "%VERIFY_SCRIPT%"
echo.>> "%VERIFY_SCRIPT%"
echo if __name__ == "__main__":>> "%VERIFY_SCRIPT%"
echo     check_pytorch_gpu()>> "%VERIFY_SCRIPT%"

echo [SUCCESS] Verification script created: %VERIFY_SCRIPT%
echo.

REM Final verification
echo Running final GPU verification...
python "%VERIFY_SCRIPT%"

echo.
echo ================================================================
echo                    INSTALLATION COMPLETE!
echo ================================================================
echo.

python -c "import torch; cuda_available = torch.cuda.is_available(); print('[SUCCESS] PyTorch with GPU support is ready!' if cuda_available else '[INFO] PyTorch installed (CPU only)'); print('GPU Status:', 'ENABLED' if cuda_available else 'DISABLED'); print('Next steps:'); print('- Your ML models can now use GPU acceleration' if cuda_available else '- Install CUDA drivers for GPU acceleration'); print('- Run verify_pytorch_gpu.py anytime to check status')"

echo.
echo Files created:
echo - %VERIFY_SCRIPT%
echo.
echo Usage in your Python code:
echo   import torch
echo   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
echo   tensor = torch.randn(100, 100).to(device)
echo.
echo ================================================================
pause
exit /b 0
