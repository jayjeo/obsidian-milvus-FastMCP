@echo off
chcp 65001 > nul
echo Applying final embeddings.py fix...
echo.

cd /d "%~dp0"
python apply_embeddings_final_fix.py

echo.
pause
