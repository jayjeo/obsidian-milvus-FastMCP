# Build script for creating the installer executable
# Run this script to create obsidian_milvus_installer.exe

import subprocess
import sys
import os
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print("PyInstaller is installed")
        return True
    except ImportError:
        print("PyInstaller is not installed")
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True

def build_installer():
    """Build the installer executable"""
    print("Building Obsidian-Milvus-FastMCP Installer...")
    
    # Get current directory
    current_dir = Path(__file__).parent
    installer_script = current_dir / "installer.py"
    
    if not installer_script.exists():
        print(f"Error: installer.py not found at {installer_script}")
        return False
    
    # PyInstaller command
    pyinstaller_args = [
        "pyinstaller",
        "--onefile",  # Single executable
        "--windowed",  # No console window
        "--name", "ObsidianMilvusInstaller",
        "--distpath", str(current_dir / "dist"),
        "--workpath", str(current_dir / "build"),
        "--specpath", str(current_dir),
        "--add-data", f"{current_dir};.",  # Include current directory
        "--hidden-import", "tkinter",
        "--hidden-import", "requests",
        "--hidden-import", "json",
        "--hidden-import", "winreg",
        "--hidden-import", "ctypes",
        "--hidden-import", "urllib",
        "--hidden-import", "threading",
        "--clean",  # Clean temporary files
        str(installer_script)
    ]
    
    # Add icon if available
    icon_path = current_dir / "installer.ico"
    if icon_path.exists():
        pyinstaller_args.insert(3, "--icon")
        pyinstaller_args.insert(4, str(icon_path))
    
    try:
        # Run PyInstaller
        subprocess.check_call(pyinstaller_args)
        
        print("\n" + "="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        print(f"Executable created at: {current_dir / 'dist' / 'ObsidianMilvusInstaller.exe'}")
        print("\nTo distribute the installer:")
        print("1. Copy ObsidianMilvusInstaller.exe from the 'dist' folder")
        print("2. Users should run it with administrator privileges")
        print("3. The installer will guide them through the setup process")
        print("="*60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main build process"""
    print("Obsidian-Milvus-FastMCP Installer Build Script")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        print("Failed to setup PyInstaller")
        sys.exit(1)
    
    # Build the installer
    if build_installer():
        print("\nBuild completed successfully!")
    else:
        print("\nBuild failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
