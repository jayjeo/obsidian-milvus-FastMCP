@echo off
setlocal enabledelayedexpansion
echo ================================================================
echo                Obsidian-Milvus FastMCP Installer & Tester
echo ================================================================
echo.

REM 현재 배치 파일이 있는 디렉토리로 이동 (사용자 독립적)
cd /d "%~dp0"

echo 📂 작업 디렉토리: %CD%
echo.

echo 1. Checking Python installation...
echo ================================================================

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python first from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version
echo.

echo 2. Checking project files...
echo ================================================================

REM Check if the MCP server file exists
if not exist "mcp_server.py" (
    echo ❌ ERROR: mcp_server.py not found in current directory
    echo Current directory: %CD%
    echo.
    echo Please make sure you are running this script from the correct project directory.
    echo The directory should contain: mcp_server.py, test_mcp.py, config.py
    echo.
    pause
    exit /b 1
)

echo ✅ mcp_server.py found

REM Check if the test file exists
if not exist "test_mcp.py" (
    echo ❌ ERROR: test_mcp.py not found in current directory
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo ✅ test_mcp.py found

REM Check if config file exists
if not exist "config.py" (
    echo ❌ ERROR: config.py not found in current directory
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo ✅ config.py found
echo.

echo 3. Checking Python dependencies...
echo ================================================================

REM Check if required packages are installed
python -c "import mcp.server.fastmcp" >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ FastMCP not found. Installing...
    pip install mcp
    if !errorlevel! neq 0 (
        echo ❌ Failed to install MCP package
        echo Please try manually: pip install mcp
        pause
        exit /b 1
    )
)

echo ✅ MCP packages available
echo.

echo 4. Testing MCP server functionality...
echo ================================================================

echo Testing MCP server with test script...
python test_mcp.py
if %errorlevel% neq 0 (
    echo ❌ MCP server test failed
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

echo ✅ MCP server test completed
echo.

echo 5. Generating Claude Desktop configuration...
echo ================================================================

REM Get current directory path and escape backslashes for JSON
set CURRENT_DIR=%CD%
set ESCAPED_DIR=!CURRENT_DIR:\=\\!

REM Set Claude Desktop config directory
set CLAUDE_CONFIG_DIR=%APPDATA%\Claude
set CLAUDE_CONFIG_FILE=%CLAUDE_CONFIG_DIR%\claude_desktop_config.json

echo Current project directory: %CURRENT_DIR%
echo Escaped for JSON: !ESCAPED_DIR!
echo Claude config location: %CLAUDE_CONFIG_FILE%
echo.

REM Create Claude config directory if it doesn't exist
if not exist "%CLAUDE_CONFIG_DIR%" (
    echo Creating Claude Desktop config directory...
    mkdir "%CLAUDE_CONFIG_DIR%"
)

REM Check if Claude Desktop config already exists
if exist "%CLAUDE_CONFIG_FILE%" (
    echo ⚠️  Claude Desktop config already exists
    echo Current config exists at: %CLAUDE_CONFIG_FILE%
    echo.
    echo Do you want to backup and update it? (Y/N)
    set /p REPLACE_CONFIG=
    if /i "!REPLACE_CONFIG!" neq "Y" (
        echo Skipping Claude Desktop config update
        goto :skip_config
    )
    
    REM Backup existing config
    for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set BACKUP_DATE=%%d-%%b-%%c
    for /f "tokens=1-3 delims=:." %%a in ("%time%") do set BACKUP_TIME=%%a-%%b-%%c
    copy "%CLAUDE_CONFIG_FILE%" "%CLAUDE_CONFIG_FILE%.backup.!BACKUP_DATE!_!BACKUP_TIME!"
    echo ✅ Backed up existing config
)

echo Generating new Claude Desktop configuration...

REM Generate the Claude Desktop config JSON - keeping existing servers and adding/updating obsidian-milvus
python -c "
import json
import os
from pathlib import Path

config_file = Path(os.environ.get('CLAUDE_CONFIG_FILE'))
current_dir = os.environ.get('CURRENT_DIR')
escaped_dir = current_dir.replace('\\', '\\\\')

# Read existing config or create new one
existing_config = {'mcpServers': {}}
if config_file.exists():
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
        if 'mcpServers' not in existing_config:
            existing_config['mcpServers'] = {}
        print(f'✅ Loaded existing config with {len(existing_config[\"mcpServers\"])} servers')
    except Exception as e:
        print(f'⚠️ Error reading existing config: {e}')
        existing_config = {'mcpServers': {}}

# Remove problematic old configurations
problematic_servers = []
for server_name, server_config in existing_config['mcpServers'].items():
    if server_name in ['milvus-obsidian'] and server_config.get('args', []) == ['-m', 'milvus_mcp.server']:
        problematic_servers.append(server_name)

for server in problematic_servers:
    del existing_config['mcpServers'][server]
    print(f'🗑️ Removed problematic config: {server}')

# Add/update the correct obsidian-milvus configuration
mcp_server_path = os.path.join(escaped_dir, 'mcp_server.py')
existing_config['mcpServers']['obsidian-milvus'] = {
    'command': 'python',
    'args': [mcp_server_path],
    'env': {
        'PYTHONPATH': escaped_dir,
        'MILVUS_HOST': 'localhost',
        'MILVUS_PORT': '19530',
        'LOG_LEVEL': 'INFO'
    }
}

# Save the configuration
try:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)
    print(f'✅ Claude Desktop configuration saved!')
    print(f'📋 Total MCP servers: {len(existing_config[\"mcpServers\"])}')
    print(f'🔧 MCP Server path: {mcp_server_path}')
except Exception as e:
    print(f'❌ Failed to save config: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ Failed to create Claude Desktop configuration
    pause
    exit /b 1
)

:skip_config
echo.

echo 6. Final verification...
echo ================================================================

echo Verifying Claude Desktop configuration...
if exist "%CLAUDE_CONFIG_FILE%" (
    echo ✅ Claude Desktop config exists
    echo.
    echo Configuration preview (obsidian-milvus section):
    echo ----------------------------------------
    python -c "
import json
import os
config_file = os.environ.get('CLAUDE_CONFIG_FILE')
try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if 'obsidian-milvus' in config.get('mcpServers', {}):
        server_config = config['mcpServers']['obsidian-milvus']
        print(f'  Command: {server_config.get(\"command\", \"\")}')
        print(f'  Script: {server_config.get(\"args\", [\"\"])[0] if server_config.get(\"args\") else \"None\"}')
        print(f'  Milvus: {server_config.get(\"env\", {}).get(\"MILVUS_HOST\", \"\")}:{server_config.get(\"env\", {}).get(\"MILVUS_PORT\", \"\")}')
    else:
        print('  ❌ obsidian-milvus server not found in config')
except Exception as e:
    print(f'  ❌ Error reading config: {e}')
"
    echo ----------------------------------------
) else (
    echo ❌ Claude Desktop config not found
)

echo.
echo Testing MCP server startup...
echo ----------------------------------------
timeout /t 2 >nul 2>nul
python -c "
import subprocess
import sys
import os

# Test if the MCP server can be imported and started
try:
    # Change to the project directory
    os.chdir(os.environ.get('CURRENT_DIR', '.'))
    
    # Try to import the main components
    import importlib.util
    
    # Check if mcp_server.py can be loaded
    spec = importlib.util.spec_from_file_location('mcp_server', 'mcp_server.py')
    if spec is None:
        print('❌ Cannot load mcp_server.py')
        sys.exit(1)
    
    print('✅ MCP server file can be loaded successfully')
    
    # Check for required imports
    with open('mcp_server.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    required_imports = [
        'from mcp.server.fastmcp import FastMCP',
        '@mcp.tool()',
        'search_documents'
    ]
    
    missing = []
    for req in required_imports:
        if req not in content:
            missing.append(req)
    
    if missing:
        print('⚠️ Some required elements may be missing:')
        for m in missing:
            print(f'   - {m}')
    else:
        print('✅ All required MCP elements found')
        
except Exception as e:
    print(f'⚠️ MCP server test warning: {e}')
"

echo.
echo ================================================================
echo                    Installation Summary
echo ================================================================
echo.
echo ✅ Python environment: OK
echo ✅ Project files: OK
echo ✅ MCP dependencies: OK
echo ✅ MCP server test: OK
echo ✅ Claude Desktop config: OK
echo.
echo 📋 Next Steps:
echo 1. Configure your Obsidian vault path in config.py:
echo    OBSIDIAN_VAULT_PATH = "C:/path/to/your/obsidian/vault"
echo 2. Run document embedding: python main.py
echo 3. Restart Claude Desktop application
echo 4. Your Obsidian-Milvus MCP server should now be available
echo 5. Try searching your Obsidian notes in Claude Desktop
echo.
echo 🔧 Configuration Details:
echo - Project Directory: %CURRENT_DIR%
echo - MCP Server: %CURRENT_DIR%\mcp_server.py
echo - Claude Config: %CLAUDE_CONFIG_FILE%
echo - Milvus Host: localhost:19530
echo.
echo ================================================================
echo                    Installation Complete!
echo ================================================================
echo.

pause