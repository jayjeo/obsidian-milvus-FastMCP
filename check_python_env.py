#!/usr/bin/env python3
"""
Python Environment Checker
Checks if all required dependencies are available and provides solutions
"""

import sys
import os
import subprocess
import importlib.util

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        if module_name in ['torch', 'torchvision']:
            # Special handling for torch modules
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return True, "Available"
            else:
                return False, "Not found"
        else:
            __import__(module_name)
            return True, "Available"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("=" * 60)
    print("PYTHON ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    # Basic Python info
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.path}")
    print("")
    
    # Required modules from requirements.txt
    required_modules = [
        'mcp',
        'fastmcp', 
        'pymilvus',
        'sentence_transformers',
        'PyPDF2',
        'markdown',  # This is the problematic one
        'beautifulsoup4',
        'python_dotenv',
        'watchdog',
        'tqdm',
        'requests',
        'psutil',
        'colorama',
        'pyyaml',
        'torch'
    ]
    
    print("CHECKING REQUIRED MODULES:")
    print("-" * 40)
    
    missing_modules = []
    for module in required_modules:
        available, status = check_module(module)
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {module:<20} - {status}")
        
        if not available:
            missing_modules.append(module)
    
    print("")
    
    # Check if pip is available
    try:
        import pip
        pip_available = True
        print("✅ pip is available")
    except ImportError:
        pip_available = False
        print("❌ pip is not available")
    
    print("")
    
    # Solution suggestions
    if missing_modules:
        print("SOLUTIONS:")
        print("-" * 40)
        print("Missing modules detected. Here are solutions:")
        print("")
        
        if pip_available:
            print("Option 1: Install missing modules with pip")
            for module in missing_modules:
                # Map some module names to their pip install names
                pip_name = module
                if module == 'python_dotenv':
                    pip_name = 'python-dotenv'
                elif module == 'beautifulsoup4':
                    pip_name = 'beautifulsoup4'
                elif module == 'PyPDF2':
                    pip_name = 'PyPDF2'
                
                print(f"  pip install {pip_name}")
        
        print("")
        print("Option 2: Install all requirements")
        print("  pip install -r requirements.txt")
        print("")
        print("Option 3: Use virtual environment")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        print("")
        
        # Create a batch file for easy installation
        batch_content = f"""@echo off
echo Installing missing Python modules...
cd /d "{os.path.dirname(os.path.abspath(__file__))}"

echo.
echo Current directory: %CD%
echo Python executable: {sys.executable}
echo.

echo Installing all requirements from requirements.txt...
"{sys.executable}" -m pip install -r requirements.txt

echo.
echo Checking installation...
"{sys.executable}" check_python_env.py

pause
"""
        
        try:
            with open("install_missing_modules.bat", "w", encoding="utf-8") as f:
                f.write(batch_content)
            print("✅ Created install_missing_modules.bat for easy installation")
        except Exception as e:
            print(f"❌ Could not create batch file: {e}")
    
    else:
        print("✅ ALL REQUIRED MODULES ARE AVAILABLE!")
        print("Your Python environment is properly configured.")
    
    print("")
    print("=" * 60)
    
    # Return status
    return len(missing_modules) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
