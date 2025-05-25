#!/usr/bin/env python3
"""
Enhanced Python Path Detection
Finds the best Python executable and creates a launcher script
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def find_python_executables():
    """Find all possible Python executables"""
    candidates = []
    
    # Current Python executable
    candidates.append(sys.executable)
    
    # Common command names
    commands = ['python', 'python3', 'py', 'python.exe', 'python3.exe']
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Get full path
                path_result = subprocess.run([cmd, '-c', 'import sys; print(sys.executable)'], 
                                           capture_output=True, text=True, timeout=5)
                if path_result.returncode == 0:
                    full_path = path_result.stdout.strip()
                    if full_path not in candidates:
                        candidates.append(full_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            continue
    
    # Common installation paths
    user_profile = os.environ.get('USERPROFILE', '')
    common_paths = [
        r'C:\Python39\python.exe',
        r'C:\Python310\python.exe', 
        r'C:\Python311\python.exe',
        r'C:\Python312\python.exe',
        r'C:\Anaconda3\python.exe',
        r'C:\Miniconda3\python.exe',
    ]
    
    if user_profile:
        common_paths.extend([
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python39\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
            f'{user_profile}\\anaconda3\\python.exe',
            f'{user_profile}\\miniconda3\\python.exe',
        ])
    
    for path in common_paths:
        if os.path.exists(path) and path not in candidates:
            candidates.append(path)
    
    return candidates

def test_python_environment(python_exe):
    """Test if Python environment has required modules"""
    try:
        # Test basic functionality
        result = subprocess.run([python_exe, '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, "Version check failed"
        
        version = result.stdout.strip()
        
        # Test critical modules
        test_script = '''
import sys
try:
    import markdown
    print("markdown: OK")
except ImportError:
    print("markdown: MISSING")
    sys.exit(1)

try:
    import PyPDF2
    print("PyPDF2: OK") 
except ImportError:
    print("PyPDF2: MISSING")

try:
    import pymilvus
    print("pymilvus: OK")
except ImportError:
    print("pymilvus: MISSING")

print("Environment test completed")
'''
        
        result = subprocess.run([python_exe, '-c', test_script], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return True, f"Working environment with {version}"
        else:
            return False, f"Missing modules: {result.stdout}"
            
    except Exception as e:
        return False, f"Error testing: {str(e)}"

def create_python_launcher():
    """Create a Python launcher that uses the best available Python"""
    script_dir = Path(__file__).parent
    
    print("Searching for Python installations...")
    candidates = find_python_executables()
    
    if not candidates:
        print("❌ No Python installations found!")
        return None
    
    print(f"Found {len(candidates)} Python candidates:")
    
    best_python = None
    working_pythons = []
    
    for python_exe in candidates:
        print(f"\nTesting: {python_exe}")
        works, message = test_python_environment(python_exe)
        
        if works:
            print(f"✅ {message}")
            working_pythons.append((python_exe, message))
            if best_python is None:
                best_python = python_exe
        else:
            print(f"❌ {message}")
    
    if not best_python:
        print("\n❌ No working Python environment found!")
        print("Please install required modules with:")
        print("  pip install -r requirements.txt")
        return None
    
    print(f"\n✅ Best Python: {best_python}")
    
    # Create launcher batch file
    launcher_content = f'''@echo off
REM Auto-generated Python launcher
REM Uses the best detected Python environment

set PYTHON_EXE="{best_python}"
set SCRIPT_DIR=%~dp0

echo Using Python: %PYTHON_EXE%
echo Script directory: %SCRIPT_DIR%

cd /d "%SCRIPT_DIR%"

REM Run the target script with detected Python
%PYTHON_EXE% %*
'''
    
    launcher_path = script_dir / "python_launcher.bat"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"✅ Created launcher: {launcher_path}")
    
    # Create VBS launcher
    vbs_content = f'''
' Auto-generated VBS launcher with detected Python
Dim objShell, scriptDir, pythonExe, targetScript

Set objShell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
pythonExe = "{best_python}"

' Change to script directory
objShell.CurrentDirectory = scriptDir

' Run the MCP server
targetScript = scriptDir & "\\temp_mcp_option1_only.py"
objShell.Run pythonExe & " """ & targetScript & """", 0, False

Set objShell = Nothing
'''
    
    vbs_launcher_path = script_dir / "python_launcher.vbs"
    with open(vbs_launcher_path, 'w') as f:
        f.write(vbs_content)
    
    print(f"✅ Created VBS launcher: {vbs_launcher_path}")
    
    # Create config file
    config = {
        "best_python": best_python,
        "working_pythons": working_pythons,
        "detected_at": str(Path(__file__).parent),
        "script_name": __file__
    }
    
    config_path = script_dir / "python_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved config: {config_path}")
    
    return best_python

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED PYTHON ENVIRONMENT DETECTION")
    print("=" * 60)
    
    result = create_python_launcher()
    
    if result:
        print(f"\n✅ Setup completed successfully!")
        print(f"Detected Python: {result}")
        print("\nYou can now use:")
        print("  python_launcher.bat <script.py>")
        print("  python_launcher.vbs (for MCP server)")
    else:
        print(f"\n❌ Setup failed!")
        print("Please install Python and required modules")
        
    print("\n" + "=" * 60)
