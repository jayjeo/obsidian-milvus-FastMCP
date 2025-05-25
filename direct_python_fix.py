#!/usr/bin/env python3
"""
Direct Python Environment Fix
Finds the Python with modules and installs missing ones
"""

import sys
import os
import subprocess
import importlib.util

def find_python_executables():
    """Find all Python executables"""
    candidates = [
        sys.executable,  # Current Python
        'python',
        'python3', 
        'py',
        'python.exe',
        'python3.exe'
    ]
    
    # Add common paths
    user_profile = os.environ.get('USERPROFILE', '')
    if user_profile:
        candidates.extend([
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python39\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe',
            f'{user_profile}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
        ])
    
    working_pythons = []
    
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Get full path
                path_result = subprocess.run([candidate, '-c', 'import sys; print(sys.executable)'], 
                                           capture_output=True, text=True, timeout=5)
                if path_result.returncode == 0:
                    full_path = path_result.stdout.strip()
                    if full_path not in [wp[0] for wp in working_pythons]:
                        working_pythons.append((full_path, candidate, result.stdout.strip()))
        except:
            continue
    
    return working_pythons

def test_modules(python_exe):
    """Test if required modules are available"""
    required_modules = ['markdown', 'PyPDF2', 'pymilvus', 'mcp', 'fastmcp']
    missing = []
    available = []
    
    for module in required_modules:
        try:
            result = subprocess.run([python_exe, '-c', f'import {module}; print("OK")'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available.append(module)
            else:
                missing.append(module)
        except:
            missing.append(module)
    
    return available, missing

def install_modules(python_exe, missing_modules):
    """Install missing modules"""
    print(f"Installing missing modules in {python_exe}...")
    
    try:
        # First try requirements.txt
        result = subprocess.run([python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Successfully installed from requirements.txt")
            return True
        else:
            print(f"❌ Requirements.txt failed: {result.stderr}")
    except:
        pass
    
    # Try individual modules
    success_count = 0
    for module in missing_modules:
        try:
            # Map module names to pip names
            pip_name = module
            if module == 'PyPDF2':
                pip_name = 'PyPDF2'
            elif module == 'markdown':
                pip_name = 'markdown>=3.4.3'
            elif module == 'pymilvus':
                pip_name = 'pymilvus>=2.3.0'
            elif module == 'mcp':
                pip_name = 'mcp>=1.2.0'
            elif module == 'fastmcp':
                pip_name = 'fastmcp>=2.0.0'
            
            print(f"Installing {pip_name}...")
            result = subprocess.run([python_exe, '-m', 'pip', 'install', pip_name], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {module} installed successfully")
                success_count += 1
            else:
                print(f"❌ Failed to install {module}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing {module}: {e}")
    
    return success_count > 0

def create_working_launcher(python_exe):
    """Create launcher with working Python"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create batch launcher
    bat_content = f'''@echo off
echo Using verified Python: {python_exe}
cd /d "{script_dir}"

REM Test environment
"{python_exe}" -c "import markdown; print('Environment OK')"
if not %errorlevel% equ 0 (
    echo ❌ Environment test failed
    pause
    exit /b 1
)

echo ✅ Environment verified - starting MCP server
"{python_exe}" mcp_server.py
pause
'''
    
    with open('start_mcp_working.bat', 'w') as f:
        f.write(bat_content)
    
    print(f"✅ Created start_mcp_working.bat")
    
    # Create VBS launcher
    vbs_content = f'''
Dim objShell
Set objShell = CreateObject("WScript.Shell")

objShell.CurrentDirectory = "{script_dir}"
objShell.Run """"{python_exe}"""" mcp_server.py", 0, False

Set objShell = Nothing
'''
    
    with open('start_mcp_working.vbs', 'w') as f:
        f.write(vbs_content)
    
    print(f"✅ Created start_mcp_working.vbs")
    
    return True

def main():
    print("=" * 60)
    print("DIRECT PYTHON ENVIRONMENT FIX")
    print("=" * 60)
    print()
    
    # Find Python executables
    print("Finding Python installations...")
    pythons = find_python_executables()
    
    if not pythons:
        print("❌ No Python installations found!")
        return False
    
    print(f"Found {len(pythons)} Python installation(s):")
    for i, (path, cmd, version) in enumerate(pythons, 1):
        print(f"  {i}. {cmd} -> {path} ({version})")
    
    print()
    
    # Test each Python for modules
    working_python = None
    
    for path, cmd, version in pythons:
        print(f"Testing {cmd} ({path})...")
        available, missing = test_modules(path)
        
        print(f"  Available: {available}")
        print(f"  Missing: {missing}")
        
        if not missing:
            print(f"✅ Perfect! All modules available in {cmd}")
            working_python = path
            break
        elif len(missing) <= 2:  # Try to install if only a few missing
            print(f"⚠️ Some modules missing, attempting installation...")
            if install_modules(path, missing):
                # Re-test after installation
                available, missing = test_modules(path)
                if not missing:
                    print(f"✅ Successfully installed missing modules in {cmd}")
                    working_python = path
                    break
                else:
                    print(f"❌ Still missing modules after installation: {missing}")
        
        print()
    
    if not working_python:
        print("❌ Could not find or create a working Python environment")
        print()
        print("Manual steps:")
        print("1. Open Command Prompt")
        print("2. Run: pip install markdown PyPDF2 pymilvus mcp fastmcp")
        print("3. Re-run this script")
        return False
    
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Working Python: {working_python}")
    print()
    
    # Create launchers
    create_working_launcher(working_python)
    
    print("You can now start the MCP server using:")
    print("  start_mcp_working.bat")
    print("  start_mcp_working.vbs")
    print()
    
    # Test final environment
    print("Final environment test...")
    try:
        result = subprocess.run([working_python, '-c', 'import markdown; import PyPDF2; print("✅ All modules OK")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Final test passed!")
            return True
        else:
            print(f"❌ Final test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Final test error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
            print("You can now run: start_mcp_working.bat")
        else:
            print("\n💥 Setup failed - manual intervention required")
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        input("Press Enter to exit...")
