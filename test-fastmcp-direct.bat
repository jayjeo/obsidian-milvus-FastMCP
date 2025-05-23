@echo off
chcp 65001 >nul
echo ================================================================
echo              FastMCP Server Direct Test
echo ================================================================
echo.

REM Change to the correct directory
g:
cd /d "G:\JJ Dropbox\J J\PythonWorks\milvus\obsidian-milvus-FastMCP"

echo Current directory: %CD%
echo.

echo [Step 1] Checking Python and required modules...
python -c "import sys; print(f'Python {sys.version}')"

echo Testing mcp module...
python -c "try:" -c "    import mcp" -c "    print('[OK] mcp module found')" -c "except Exception as e:" -c "    print(f'[ERROR] mcp module missing: {e}')"

echo Testing FastMCP module...
python -c "try:" -c "    from mcp.server.fastmcp import FastMCP" -c "    print('[OK] FastMCP module found')" -c "except Exception as e:" -c "    print(f'[ERROR] FastMCP module issue: {e}')"

echo.

echo [Step 2] Checking config file...
if exist "config.py" (
    echo [OK] config.py found
) else (
    echo [ERROR] config.py missing
    pause
    exit /b 1
)
echo.

echo [Step 3] Testing Milvus connection...
python -c "try:" -c "    import config" -c "    from milvus_manager import MilvusManager" -c "    print('[TEST] Initializing Milvus manager...')" -c "    manager = MilvusManager()" -c "    print('[OK] Milvus connection successful')" -c "except Exception as e:" -c "    print(f'[WARNING] Milvus connection issue: {e}')" -c "    print('Please check if Milvus is running.')"

echo.

echo [Step 4] Starting FastMCP server directly...
echo ================================================================
echo NOTE: Server will start. Use Ctrl+C to stop.
echo ================================================================
echo.

timeout /t 3 /nobreak >nul

echo Starting FastMCP server...
python mcp_server.py

echo.
echo ================================================================
echo                Server Test Complete
echo ================================================================
pause
