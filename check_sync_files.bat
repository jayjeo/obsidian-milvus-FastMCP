@echo off
chcp 65001 > nul
echo Checking and syncing required files...
echo.

cd /d "%~dp0"
python sync_files.py

echo.
pause
