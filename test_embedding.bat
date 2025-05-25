@echo off
chcp 65001 > nul
echo Running Embedding Model Loading Test...
echo.

cd /d "%~dp0"
python test_embedding_load.py

echo.
pause
