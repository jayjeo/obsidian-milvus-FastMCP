# Build script for creating the installer executable
# Run this script to create ObsidianMilvusInstaller.exe

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
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install PyInstaller: {e}")
            return False

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
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        "--hidden-import", "requests",
        "--hidden-import", "json",
        "--hidden-import", "winreg",
        "--hidden-import", "ctypes",
        "--hidden-import", "urllib",
        "--hidden-import", "urllib.request",
        "--hidden-import", "threading",
        "--hidden-import", "subprocess",
        "--hidden-import", "pathlib",
        "--hidden-import", "shutil",
        "--hidden-import", "time",
        "--hidden-import", "os",
        "--hidden-import", "sys",
        "--clean",  # Clean temporary files
        "--noconfirm",  # Overwrite output directory without confirmation
        str(installer_script)
    ]
    
    # Add icon if available
    icon_path = current_dir / "installer.ico"
    if icon_path.exists():
        pyinstaller_args.insert(4, "--icon=" + str(icon_path))
        print(f"Using icon: {icon_path}")
    
    try:
        # Create directories if they don't exist
        (current_dir / "dist").mkdir(exist_ok=True)
        (current_dir / "build").mkdir(exist_ok=True)
        
        # Run PyInstaller
        print("\nRunning PyInstaller with the following arguments:")
        print(" ".join(pyinstaller_args[:10]) + "...")  # Show first few args
        
        result = subprocess.run(pyinstaller_args, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("BUILD SUCCESSFUL!")
            print("="*60)
            
            exe_path = current_dir / "dist" / "ObsidianMilvusInstaller.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"Executable created: {exe_path}")
                print(f"File size: {size_mb:.2f} MB")
            
            print("\nTo distribute the installer:")
            print("1. Copy ObsidianMilvusInstaller.exe from the 'dist' folder")
            print("2. Users should run it with administrator privileges")
            print("3. The installer will guide them through the setup process")
            print("="*60)
            
            return True
        else:
            print(f"\nBuild failed with return code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        if hasattr(e, 'output'):
            print(f"Output: {e.output}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def clean_build_artifacts():
    """Clean up previous build artifacts"""
    print("Cleaning up previous build artifacts...")
    current_dir = Path(__file__).parent
    
    # Directories to clean
    dirs_to_clean = ["build", "__pycache__"]
    for dir_name in dirs_to_clean:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            try:
                import shutil
                shutil.rmtree(dir_path)
                print(f"Removed {dir_path}")
            except Exception as e:
                print(f"Warning: Could not remove {dir_path}: {e}")
    
    # Files to clean
    files_to_clean = ["*.spec", "*.log"]
    for pattern in files_to_clean:
        for file_path in current_dir.glob(pattern):
            try:
                file_path.unlink()
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")

def main():
    """Main build process"""
    print("Obsidian-Milvus-FastMCP Installer Build Script")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    print(f"Running from: {Path(__file__).parent}")
    
    # Clean previous build artifacts
    clean_build_artifacts()
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        print("Failed to setup PyInstaller")
        print("Please install it manually: pip install pyinstaller")
        sys.exit(1)
    
    # Build the installer
    print("\n" + "-"*60)
    if build_installer():
        print("\nBuild completed successfully!")
        print("You can now distribute the ObsidianMilvusInstaller.exe file")
    else:
        print("\nBuild failed!")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()