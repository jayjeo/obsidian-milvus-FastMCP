@echo off
echo ============================================================
echo Python Environment Diagnostic and Fix Tool
echo ============================================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Step 1: Testing Python installations...
echo --------------------------------------------------------

echo Testing 'python' command...
python --version 2>nul
if %errorlevel% equ 0 (
    echo ✅ 'python' command works
    set PYTHON_CMD=python
    goto test_modules
) else (
    echo ❌ 'python' command failed
)

echo Testing 'python3' command...
python3 --version 2>nul
if %errorlevel% equ 0 (
    echo ✅ 'python3' command works
    set PYTHON_CMD=python3
    goto test_modules
) else (
    echo ❌ 'python3' command failed
)

echo Testing 'py' command...
py --version 2>nul
if %errorlevel% equ 0 (
    echo ✅ 'py' command works
    set PYTHON_CMD=py
    goto test_modules
) else (
    echo ❌ 'py' command failed
)

echo.
echo ❌ No working Python command found!
echo Please install Python from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
pause
exit /b 1

:test_modules
echo.
echo Step 2: Testing required modules...
echo --------------------------------------------------------
echo Using Python command: %PYTHON_CMD%
echo.

echo Running detailed environment check...
%PYTHON_CMD% check_python_env.py
if %errorlevel% equ 0 (
    echo ✅ All modules are available
    goto test_mcp_startup
) else (
    echo ❌ Some modules are missing
    goto install_modules
)

:install_modules
echo.
echo Step 3: Installing missing modules...
echo --------------------------------------------------------
echo This may take a few minutes...
echo.

echo Installing from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo ✅ Requirements installation completed
) else (
    echo ❌ Requirements installation failed
    echo Trying individual module installation...
    
    echo Installing markdown module...
    %PYTHON_CMD% -m pip install markdown>=3.4.3
    
    echo Installing other critical modules...
    %PYTHON_CMD% -m pip install PyPDF2>=3.0.1
    %PYTHON_CMD% -m pip install beautifulsoup4>=4.12.2
    %PYTHON_CMD% -m pip install python-dotenv>=1.0.0
)

echo.
echo Re-testing modules after installation...
%PYTHON_CMD% check_python_env.py
if %errorlevel% equ 0 (
    echo ✅ Module installation successful
) else (
    echo ❌ Some modules still missing
    echo.
    echo Manual installation may be required.
    echo Try running these commands manually:
    echo   %PYTHON_CMD% -m pip install --upgrade pip
    echo   %PYTHON_CMD% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:test_mcp_startup
echo.
echo Step 4: Testing MCP server startup...
echo --------------------------------------------------------

echo Creating test script...
echo import sys; print("Python OK"); import markdown; print("markdown OK") > test_imports.py

%PYTHON_CMD% test_imports.py
if %errorlevel% equ 0 (
    echo ✅ Import test successful
    del test_imports.py
) else (
    echo ❌ Import test failed
    del test_imports.py
    pause
    exit /b 1
)

echo.
echo ✅ Environment diagnostic completed successfully!
echo Your Python environment should now work with the MCP server.
echo.
echo You can now try running:
echo   auto_start_mcp_server_enhanced.vbs
echo.
pause
