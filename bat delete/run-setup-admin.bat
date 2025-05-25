@echo off
echo Starting admin setup script...
cd /d "g:\JJ Dropbox\J J\PythonWorks\milvus\obsidian-milvus-FastMCP"
echo Current directory: %CD%
echo Checking if setup.py exists...
if exist "setup.py" (
    echo setup.py found, proceeding...
) else (
    echo ERROR: setup.py not found in %CD%
    echo This is unexpected as we should be in the correct directory.
)
echo Running run-setup.bat...
call run-setup.bat
echo Script finished.
pause
