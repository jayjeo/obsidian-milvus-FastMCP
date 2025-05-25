# Installer Build Instructions

This directory contains the installer for Obsidian-Milvus-FastMCP.

## Files

- `installer.py` - Main installer application with GUI
- `build_installer.py` - Basic build script
- `build_advanced.py` - Advanced build script with Windows manifest
- `installer_state.json` - Created during installation to track progress (auto-generated)

## Building the Installer

### Prerequisites

1. Python 3.7 or higher
2. pip (Python package manager)

### Build Steps

1. Open a command prompt in this directory
2. Run one of the build scripts:

   **Basic build:**
   ```
   python build_installer.py
   ```

   **Advanced build (recommended):**
   ```
   python build_advanced.py
   ```

3. The executable will be created in the `dist` folder

### What the Installer Does

The installer automates the entire setup process:

1. **Checks for Conda** - Verifies Anaconda/Miniconda is installed
2. **Clones Repository** - Downloads the project from GitHub
3. **Installs Dependencies** - Sets up all required Python packages
4. **Installs Podman** - Container runtime for Milvus
5. **Configures WSL** - Sets up Windows Subsystem for Linux
6. **Installs Ubuntu** - Downloads and installs Ubuntu for WSL
7. **Configures Paths** - Updates config.py with correct paths
8. **Initializes Containers** - Sets up Podman and Milvus
9. **Guides Manual Steps** - Helps configure auto-startup
10. **Completes Setup** - Finalizes the installation

### Features

- **Resume Support** - Can resume installation after system restarts
- **Progress Tracking** - Shows current step and overall progress
- **Error Handling** - Provides clear error messages
- **Admin Detection** - Checks for required privileges
- **Logging** - Detailed installation log

### Distribution

To distribute the installer:

1. Build using `build_advanced.py`
2. Find the package in `dist/ObsidianMilvusInstaller_v1.0.0/`
3. Share the entire folder or just the .exe file
4. Users must run as Administrator

### Troubleshooting

If the build fails:

1. Ensure PyInstaller is installed: `pip install pyinstaller`
2. Check for antivirus interference
3. Try building with console mode first (remove `--windowed`)
4. Check the build logs in the `build` directory

### Testing

To test the installer without building:

```
python installer.py
```

This runs the installer directly from Python.
