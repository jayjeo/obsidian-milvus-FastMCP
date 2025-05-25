@echo off
chcp 65001 > nul
echo Applying embeddings.py fix...
echo.

cd /d "%~dp0"
python fix_embeddings.py

echo.
echo Fix applied! Now run start_mcp_with_encoding_fix.bat
echo.
pause
