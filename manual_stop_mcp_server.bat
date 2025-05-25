@echo off
echo Stopping MCP Server...
echo This will close all Python processes running main.py
pause
taskkill /f /im python.exe /fi "WINDOWTITLE eq main.py*" 2>nul
taskkill /f /im python3.exe /fi "WINDOWTITLE eq main.py*" 2>nul
taskkill /f /im py.exe /fi "WINDOWTITLE eq main.py*" 2>nul
echo MCP Server stopped
pause
