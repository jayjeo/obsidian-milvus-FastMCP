# Advanced installer build configuration
# This creates a more polished installer with better error handling

import subprocess
import sys
import os
import shutil
from pathlib import Path

# Configuration
INSTALLER_NAME = "ObsidianMilvusInstaller"
VERSION = "1.0.0"
DESCRIPTION = "Automated installer for Obsidian-Milvus-FastMCP integration"
COMPANY = "Obsidian-Milvus Project"

def create_version_file():
    """Create version information file for Windows"""
    version_content = f'''
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
        [StringStruct(u'CompanyName', u'{COMPANY}'),
        StringStruct(u'FileDescription', u'{DESCRIPTION}'),
        StringStruct(u'FileVersion', u'{VERSION}'),
        StringStruct(u'InternalName', u'{INSTALLER_NAME}'),
        StringStruct(u'LegalCopyright', u'Copyright 2024'),
        StringStruct(u'OriginalFilename', u'{INSTALLER_NAME}.exe'),
        StringStruct(u'ProductName', u'{INSTALLER_NAME}'),
        StringStruct(u'ProductVersion', u'{VERSION}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    version_file = Path("version_info.txt")
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
    
    manifest_file = Path("installer.manifest")
    with open(manifest_file, 'w') as f:
        f.write(manifest_content)
    
    return manifest_file

def build_advanced_installer():
    """Build installer with advanced options"""
    print(f"Building {INSTALLER_NAME} v{VERSION}")
    print("="*60)
    
    current_dir = Path(__file__).parent
    installer_script = current_dir / "installer.py"
    
    # Create version file
    version_file = create_version_file()
    
    # Create manifest
    manifest_file = create_manifest()
    
    # Prepare build command
    pyinstaller_args = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", INSTALLER_NAME,
        "--distpath", str(current_dir / "dist"),
        "--workpath", str(current_dir / "build"),
        "--specpath", str(current_dir),
        "--version-file", str(version_file),
        "--manifest", str(manifest_file),
        "--uac-admin",  # Request admin privileges
        "--add-data", f"{current_dir};.",
        # Hidden imports
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        "--hidden-import", "requests",
        "--hidden-import", "json",
        "--hidden-import", "winreg",
        "--hidden-import", "ctypes",
        "--hidden-import", "urllib.request",
        "--hidden-import", "threading",
        "--hidden-import", "subprocess",
        "--hidden-import", "pathlib",
        # Optimizations
        "--clean",
        "--noconfirm",
        str(installer_script)
    ]
    
    try:
        # Run PyInstaller
        subprocess.check_call(pyinstaller_args)
        
        # Clean up temporary files
        version_file.unlink(missing_ok=True)
        manifest_file.unlink(missing_ok=True)
        
        # Create distribution package
        dist_dir = current_dir / "dist"
        package_dir = dist_dir / f"{INSTALLER_NAME}_v{VERSION}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy executable
        exe_source = dist_dir / f"{INSTALLER_NAME}.exe"
        exe_dest = package_dir / f"{INSTALLER_NAME}.exe"
        shutil.copy2(exe_source, exe_dest)
        
        # Create README
        readme_content = f"""
{INSTALLER_NAME}