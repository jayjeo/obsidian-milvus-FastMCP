@echo off
echo ============================================================
echo OBSIDIAN-MILVUS-MCP ENVIRONMENT SETUP AND DIAGNOSTICS
echo ============================================================
echo.
echo This script will:
echo 1. Detect Python installations
echo 2. Check for required modules 
echo 3. Install missing modules
echo 4. Create optimized launcher scripts
echo 5. Test MCP server startup
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM Step 1: Check if Python detection script exists
if not exist "detect_python_environment.py" (
    echo ❌ detect_python_environment.py not found
    echo Please ensure all files are in the correct directory
    pause
    exit /b 1
)

if not exist "check_python_env.py" (
    echo ❌ check_python_env.py not found
    echo Please ensure all files are in the correct directory
    pause
    exit /b 1
)

echo Step 1: Detecting Python environments...
echo --------------------------------------------------------

REM Try to run Python detection with different commands
python detect_python_environment.py 2>nul
if %errorlevel% equ 0 goto check_result

python3 detect_python_environment.py 2>nul
if %errorlevel% equ 0 goto check_result

py detect_python_environment.py 2>nul
if %errorlevel% equ 0 goto check_result

echo ❌ Could not run Python detection script
echo Please ensure Python is installed and accessible
echo.
echo Try installing Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:check_result
echo.
echo Step 2: Checking detection results...
echo --------------------------------------------------------

if exist "python_config.json" (
    echo ✅ Python environment detected successfully
    if exist "python_launcher.bat" (
        echo ✅ Batch launcher created
    )
    if exist "python_launcher.vbs" (
        echo ✅ VBS launcher created
    )
) else (
    echo ❌ Python detection failed
    echo Running detailed environment check...
    
    REM Try basic environment check
    python check_python_env.py 2>nul
    if not %errorlevel% equ 0 (
        python3 check_python_env.py 2>nul
        if not %errorlevel% equ 0 (
            py check_python_env.py 2>nul
        )
    )
    
    echo.
    echo Trying manual module installation...
    echo --------------------------------------------------------
    
    REM Try to install requirements
    python -m pip install -r requirements.txt 2>nul
    if not %errorlevel% equ 0 (
        python3 -m pip install -r requirements.txt 2>nul
        if not %errorlevel% equ 0 (
            py -m pip install -r requirements.txt 2>nul
        )
    )
    
    REM Re-run detection after installation
    python detect_python_environment.py 2>nul
    if not %errorlevel% equ 0 (
        python3 detect_python_environment.py 2>nul
        if not %errorlevel% equ 0 (
            py detect_python_environment.py 2>nul
        )
    )
)

echo.
echo Step 3: Testing MCP server startup...
echo --------------------------------------------------------

if exist "python_launcher.bat" (
    echo Testing batch launcher...
    echo import sys; print("✅ Python launcher test successful"); import markdown; print("✅ markdown module available") > test_launcher.py
    
    call python_launcher.bat test_launcher.py
    if %errorlevel% equ 0 (
        echo ✅ Batch launcher works correctly
    ) else (
        echo ❌ Batch launcher test failed
    )
    
    del test_launcher.py 2>nul
) else (
    echo ❌ No launcher created
)

echo.
echo Step 4: Creating backup startup methods...
echo --------------------------------------------------------

REM Create a simple startup batch file as backup
echo @echo off > start_mcp_simple.bat
echo cd /d "%%~dp0" >> start_mcp_simple.bat
if exist "python_launcher.bat" (
    echo call python_launcher.bat temp_mcp_option1_only.py >> start_mcp_simple.bat
) else (
    echo REM Try different Python commands >> start_mcp_simple.bat
    echo python temp_mcp_option1_only.py >> start_mcp_simple.bat
    echo if not %%errorlevel%% equ 0 python3 temp_mcp_option1_only.py >> start_mcp_simple.bat
    echo if not %%errorlevel%% equ 0 py temp_mcp_option1_only.py >> start_mcp_simple.bat
)
echo pause >> start_mcp_simple.bat

echo ✅ Created start_mcp_simple.bat as backup

echo.
echo ============================================================
echo SETUP SUMMARY
echo ============================================================

if exist "python_config.json" (
    echo ✅ Environment detection: SUCCESS
    
    REM Display detected Python
    if exist "python_launcher.bat" (
        echo ✅ Optimized launchers: CREATED
        echo.
        echo You can now use these methods to start the MCP server:
        echo.
        echo Method 1 (Recommended): Enhanced VBS launcher
        echo   auto_start_mcp_server_enhanced.vbs
        echo.
        echo Method 2: Optimized VBS launcher  
        echo   python_launcher.vbs
        echo.
        echo Method 3: Batch launcher
        echo   start_mcp_simple.bat
        echo.
        echo Method 4: Direct batch
        echo   python_launcher.bat temp_mcp_option1_only.py
    ) else (
        echo ⚠️ Launchers: PARTIAL (detection succeeded but launcher creation failed)
        echo.
        echo You can try these methods:
        echo.
        echo Method 1: Enhanced VBS (may work)
        echo   auto_start_mcp_server_enhanced.vbs
        echo.  
        echo Method 2: Simple batch
        echo   start_mcp_simple.bat
    )
) else (
    echo ❌ Environment detection: FAILED
    echo.
    echo Manual troubleshooting required:
    echo.
    echo 1. Install Python from https://www.python.org/downloads/
    echo 2. Add Python to PATH during installation
    echo 3. Run: pip install -r requirements.txt
    echo 4. Re-run this setup script
)

echo.
echo For troubleshooting, check these log files:
echo   - vbs_startup.log
echo   - vbs_debug.log  
echo   - auto_startup_mcp.log
echo.
echo ============================================================
pause
