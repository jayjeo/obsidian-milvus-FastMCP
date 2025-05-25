@echo off
echo ============================================================
echo PYTHON ENVIRONMENT SYNCHRONIZATION TOOL
echo ============================================================
echo.
echo This tool will identify and fix Python environment issues
echo between different Python installations on your system.
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Step 1: Detecting all Python installations...
echo --------------------------------------------------------

setlocal enabledelayedexpansion

REM Array to store Python paths
set pythonCount=0

REM Test common Python commands
for %%p in (python python3 py) do (
    %%p --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Found working command: %%p
        
        REM Get the actual executable path
        for /f "tokens=*" %%i in ('%%p -c "import sys; print(sys.executable)"') do (
            set /a pythonCount+=1
            set python!pythonCount!=%%i
            set command!pythonCount!=%%p
            echo   Path: %%i
        )
    ) else (
        echo ❌ Command not working: %%p
    )
)

echo.
echo Found %pythonCount% Python installation(s)
echo.

if %pythonCount% equ 0 (
    echo ❌ No Python installations found!
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Step 2: Testing each Python for required modules...
echo --------------------------------------------------------

set workingPython=
set workingCommand=

for /l %%i in (1,1,%pythonCount%) do (
    echo.
    echo Testing Python %%i: !command%%i! ^(!python%%i!^)
    echo --------------------------------------------------------
    
    REM Test markdown module
    !command%%i! -c "import markdown; print('✅ markdown:', markdown.__version__)" 2>nul
    if !errorlevel! equ 0 (
        echo ✅ markdown module: OK
        set markdownOK=1
    ) else (
        echo ❌ markdown module: MISSING
        set markdownOK=0
    )
    
    REM Test other critical modules
    !command%%i! -c "import PyPDF2; print('✅ PyPDF2: OK')" 2>nul
    if !errorlevel! equ 0 (
        echo ✅ PyPDF2 module: OK
        set pdfOK=1
    ) else (
        echo ❌ PyPDF2 module: MISSING
        set pdfOK=0
    )
    
    REM If this Python has all modules, use it
    if !markdownOK! equ 1 if !pdfOK! equ 1 (
        echo ✅ This Python has all required modules!
        set workingPython=!python%%i!
        set workingCommand=!command%%i!
        goto found_working
    )
)

echo.
echo ❌ No Python installation has all required modules!
echo.

REM Find the best candidate for installation
echo Step 3: Installing missing modules...
echo --------------------------------------------------------

for /l %%i in (1,1,%pythonCount%) do (
    echo.
    echo Trying to install modules in Python %%i: !command%%i!
    echo --------------------------------------------------------
    
    REM Try to upgrade pip first
    !command%%i! -m pip install --upgrade pip
    
    REM Install requirements
    !command%%i! -m pip install -r requirements.txt
    
    REM Test if installation worked
    !command%%i! -c "import markdown; import PyPDF2; print('✅ All modules installed successfully')" 2>nul
    if !errorlevel! equ 0 (
        echo ✅ Successfully installed modules in this Python!
        set workingPython=!python%%i!
        set workingCommand=!command%%i!
        goto found_working
    ) else (
        echo ❌ Installation failed in this Python
    )
)

echo.
echo ❌ Could not install modules in any Python installation!
echo.
echo Manual steps required:
echo 1. Open Command Prompt as Administrator
echo 2. Run: pip install -r requirements.txt
echo 3. If that fails, try: python -m pip install markdown PyPDF2 pymilvus
echo.
pause
exit /b 1

:found_working
echo.
echo ============================================================
echo SUCCESS - WORKING PYTHON FOUND
echo ============================================================
echo.
echo Working Python: %workingCommand%
echo Full path: %workingPython%
echo.

echo Step 4: Creating environment-specific launchers...
echo --------------------------------------------------------

REM Create Python launcher with the working Python
echo @echo off > python_working.bat
echo REM Auto-generated launcher using verified working Python >> python_working.bat
echo set PYTHON_EXE="%workingPython%" >> python_working.bat
echo set SCRIPT_DIR=%%~dp0 >> python_working.bat
echo cd /d "%%SCRIPT_DIR%%" >> python_working.bat
echo %%PYTHON_EXE%% %%* >> python_working.bat

echo ✅ Created python_working.bat

REM Create VBS launcher with the working Python
echo ' Auto-generated VBS launcher with verified working Python > python_working.vbs
echo Dim objShell, scriptDir, pythonExe >> python_working.vbs
echo Set objShell = CreateObject("WScript.Shell") >> python_working.vbs
echo scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) >> python_working.vbs
echo pythonExe = "%workingPython%" >> python_working.vbs
echo objShell.CurrentDirectory = scriptDir >> python_working.vbs
echo objShell.Run pythonExe ^& " " ^& WScript.Arguments(0), 0, False >> python_working.vbs
echo Set objShell = Nothing >> python_working.vbs

echo ✅ Created python_working.vbs

REM Create MCP startup script using working Python
echo @echo off > start_mcp_verified.bat
echo echo Starting MCP Server with verified Python environment... >> start_mcp_verified.bat
echo cd /d "%%~dp0" >> start_mcp_verified.bat
echo echo Using Python: %workingPython% >> start_mcp_verified.bat
echo echo. >> start_mcp_verified.bat
echo REM Test environment first >> start_mcp_verified.bat
echo "%workingPython%" -c "import markdown; print('✅ Environment verified')" >> start_mcp_verified.bat
echo if not %%errorlevel%% equ 0 ( >> start_mcp_verified.bat
echo     echo ❌ Environment verification failed >> start_mcp_verified.bat
echo     pause >> start_mcp_verified.bat
echo     exit /b 1 >> start_mcp_verified.bat
echo ^) >> start_mcp_verified.bat
echo. >> start_mcp_verified.bat
echo echo Starting MCP server... >> start_mcp_verified.bat
echo "%workingPython%" mcp_server.py >> start_mcp_verified.bat
echo echo. >> start_mcp_verified.bat
echo echo MCP Server exited >> start_mcp_verified.bat
echo pause >> start_mcp_verified.bat

echo ✅ Created start_mcp_verified.bat

echo.
echo Step 5: Testing the working environment...
echo --------------------------------------------------------

echo Testing environment verification...
"%workingPython%" -c "import markdown; import PyPDF2; import pymilvus; print('✅ All critical modules verified')"
if %errorlevel% equ 0 (
    echo ✅ Environment test passed!
) else (
    echo ❌ Environment test failed even with working Python
    echo Something may be wrong with the module installations
)

echo.
echo ============================================================
echo SETUP COMPLETED SUCCESSFULLY
echo ============================================================
echo.
echo Working Python identified and configured:
echo   Command: %workingCommand%
echo   Path: %workingPython%
echo.
echo You can now use these verified startup methods:
echo.
echo 1. Verified MCP startup (Recommended):
echo    start_mcp_verified.bat
echo.
echo 2. Working Python launcher:
echo    python_working.bat ^<script.py^>
echo.
echo 3. Working VBS launcher:
echo    python_working.vbs ^<script.py^>
echo.
echo 4. Direct command:
echo    "%workingPython%" mcp_server.py
echo.
echo All launchers use the verified Python environment.
echo ============================================================

echo.
echo Would you like to start the MCP server now? (Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    echo.
    echo Starting MCP server with verified environment...
    call start_mcp_verified.bat
) else (
    echo.
    echo Setup completed. You can start the server later using:
    echo start_mcp_verified.bat
)

pause
