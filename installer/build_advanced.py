# Build script for creating the complete installer executable
# Run this script to create ObsidianMilvusInstaller.exe with all required features

import subprocess
import sys
import os
from pathlib import Path
import shutil

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller is available")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install PyInstaller: {e}")
            return False

def create_version_info():
    """Create version information file for Windows executable"""
    version_content = '''
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Obsidian-Milvus Project'),
        StringStruct(u'FileDescription', u'Obsidian-Milvus-FastMCP Installer'),
        StringStruct(u'FileVersion', u'1.0.0'),
        StringStruct(u'InternalName', u'ObsidianMilvusInstaller'),
        StringStruct(u'LegalCopyright', u'Copyright 2024'),
        StringStruct(u'OriginalFilename', u'ObsidianMilvusInstaller.exe'),
        StringStruct(u'ProductName', u'Obsidian-Milvus-FastMCP Installer'),
        StringStruct(u'ProductVersion', u'1.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    current_dir = Path(__file__).parent
    version_file = current_dir / "version_info.txt"
    with open(version_file, 'w') as f:
        f.write(version_content)
    
    return version_file

def create_manifest():
    """Create Windows manifest for admin privileges"""
    manifest_content = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity
    version="1.0.0.0"
    processorArchitecture="*"
    name="ObsidianMilvusInstaller"
    type="win32"
  />
  <description>Obsidian-Milvus-FastMCP Installer</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="requireAdministrator" uiAccess="false" />
      </requestedPrivileges>
    </security>
  </trustInfo>
</assembly>
'''
    
    current_dir = Path(__file__).parent
    manifest_file = current_dir / "installer.manifest"
    with open(manifest_file, 'w') as f:
        f.write(manifest_content)
    
    return manifest_file

def create_release_package(current_dir, exe_path):
    """Create a release package with documentation"""
    package_dir = current_dir / "release"
    package_dir.mkdir(exist_ok=True)
    
    # Copy executable to release folder
    release_exe = package_dir / "ObsidianMilvusInstaller.exe"
    shutil.copy2(exe_path, release_exe)
    
    # Create README for users
    readme_content = """# Obsidian-Milvus-FastMCP Installer

## Installation Instructions

1. **Right-click** on ObsidianMilvusInstaller.exe
2. Select **"Run as administrator"** (Required for installation)
3. Follow the installation wizard steps
4. The installer will guide you through:
   - Conda verification
   - Repository download
   - Dependencies installation
   - Podman and WSL setup
   - Configuration
   - System integration

## System Requirements

- Windows 10 or later
- Anaconda or Miniconda (must be pre-installed)
- 15 GB free disk space
- Internet connection
- Administrator privileges

## Pre-Installation

**IMPORTANT**: Install Anaconda before running this installer:
1. Download from: https://www.anaconda.com/download
2. Install with default settings
3. Make sure to add Anaconda to PATH

## What This Installer Does

- Clones the Obsidian-Milvus-FastMCP repository
- Installs Python dependencies via Conda
- Sets up Podman container runtime
- Configures Windows Subsystem for Linux (WSL)
- Initializes Milvus vector database
- Configures Claude Desktop integration
- Sets up automatic startup services

## After Installation

1. Restart your computer
2. Open Claude Desktop
3. Start using Obsidian semantic search!

## Support

For issues or questions, visit:
https://github.com/jayjeo/obsidian-milvus-FastMCP

## Version

Installer Version: 1.0.0
Project: Obsidian-Milvus-FastMCP
Date: 2024

"""
    
    readme_file = package_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\nRelease package created: {package_dir}")
    print(f"- Executable: {release_exe}")
    print(f"- Documentation: {readme_file}")

def build_installer():
    """Build the complete installer executable"""
    print("Building Obsidian-Milvus-FastMCP Installer...")
    print("="*60)
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Use the complete installer script
    installer_script = current_dir / "installer_ui.py"
    
    if not installer_script.exists():
        print(f"Error: {installer_script} not found!")
        print("Please make sure installer_ui.py exists in the installer directory.")
        return False
    
    # Create version and manifest files
    print("Creating version information...")
    version_file = create_version_info()
    
    print("Creating Windows manifest...")
    manifest_file = create_manifest()
    
    # PyInstaller command with all options
    pyinstaller_args = [
        "pyinstaller",
        "--onefile",                    # Single executable
        "--windowed",                   # No console window
        "--name", "Obsidian_Milvus_Installer_AMD64",
        "--distpath", str(current_dir / "dist"),
        "--workpath", str(current_dir / "build"),
        "--specpath", str(current_dir),
        "--version-file", str(version_file),
        "--manifest", str(manifest_file),
        "--uac-admin",                  # Request administrator privileges
        "--add-data", f"{current_dir};.",  # Include current directory
        # Hidden imports for all required modules
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
        "--hidden-import", "webbrowser",
        # Optimization flags
        "--clean",                      # Clean temporary files
        "--noconfirm",                  # Overwrite output without confirmation
        str(installer_script)
    ]
    
    # Add icon if available
    icon_path = current_dir / "installer.ico"
    if icon_path.exists():
        pyinstaller_args.insert(-1, "--icon=" + str(icon_path))
        print(f"Using icon: {icon_path}")
    
    try:
        # Create directories if they don't exist
        (current_dir / "dist").mkdir(exist_ok=True)
        (current_dir / "build").mkdir(exist_ok=True)
        
        # Run PyInstaller
        print("Running PyInstaller...")
        print(f"Script: {installer_script}")
        
        result = subprocess.run(pyinstaller_args, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("BUILD SUCCESSFUL!")
            print("="*60)
            
            exe_path = current_dir / "dist" / "Obsidian_Milvus_Installer_AMD64.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"Executable created: {exe_path}")
                print(f"File size: {size_mb:.2f} MB")
                
                # Create release package
                create_release_package(current_dir, exe_path)
            
            print("\nInstaller Features:")
            print("✓ Complete installation workflow")
            print("✓ Conda prerequisite check")
            print("✓ Automatic repository cloning")
            print("✓ Python dependencies installation")
            print("✓ Podman container runtime setup")
            print("✓ WSL configuration")
            print("✓ System restart handling")
            print("✓ Configuration management")
            print("✓ Error handling and recovery")
            print("✓ Progress tracking and logging")
            print("✓ Administrator privileges")
            
            print("\nTo distribute the installer:")
            print("1. Copy ObsidianMilvusInstaller.exe from the 'release' folder")
            print("2. Include the README.md file for user instructions")
            print("3. Users should run as administrator")
            print("4. Ensure users have Anaconda pre-installed")
            print("="*60)
            
            return True
        else:
            print(f"\nBuild failed with return code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
        
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary files
        try:
            version_file.unlink(missing_ok=True)
            manifest_file.unlink(missing_ok=True)
        except:
            pass

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
                shutil.rmtree(dir_path)
                print(f"Removed {dir_path}")
            except Exception as e:
                print(f"Warning: Could not remove {dir_path}: {e}")
    
    # Files to clean
    files_to_clean = ["*.spec", "*.log", "version_info.txt", "installer.manifest"]
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
    
    # Check and install requirements
    if not check_requirements():
        print("Failed to setup requirements")
        print("Please install PyInstaller manually: pip install pyinstaller")
        sys.exit(1)
    
    # Build the installer
    print("\n" + "-"*60)
    if build_installer():
        print("\nBuild completed successfully!")
        print("You can now distribute the ObsidianMilvusInstaller.exe file")
        print("Users should run it as administrator with Anaconda pre-installed")
    else:
        print("\nBuild failed!")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
