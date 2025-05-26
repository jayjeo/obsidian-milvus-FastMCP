# Configuration file for Obsidian-Milvus-FastMCP Installer
# This file contains all configuration variables and paths used throughout the installer

import os
from pathlib import Path

# Base directory - current installer directory
BASE_DIR = Path(__file__).parent.absolute()

# CUDA Toolkit download URLs
CUDA_TOOLKIT_URLS = {
    "windows_10": "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe",
    "windows_11": "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe"
}

# Download directory for installers
DOWNLOAD_DIR = BASE_DIR / "downloads"

# Temporary files directory
TEMP_DIR = BASE_DIR / "temp"

# Batch files paths
CUDA_CHECKER_SCRIPT = BASE_DIR / "CUDA_Toolkit_Checker.py"
NUMPY_FIX_SCRIPT = BASE_DIR / "fix_numpy_compatibility.bat"

# PyTorch installation commands
PYTORCH_COMMANDS = [
    "pip uninstall torch torchvision torchaudio -y",
    "conda create -n pytorch-gpu python=3.11 -y",
    "conda activate pytorch-gpu",
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    'python -c "import torch; print(\'PyTorch:\', torch.__version__); print(\'CUDA:\', torch.cuda.is_available())"'
]

# Create necessary directories
DOWNLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
