# Configuration file for conda/mamba paths
# Edit these paths according to your system setup

# Conda/Mamba Path Configuration
# ===============================

# Option 1: Use automatic detection (recommended)
# Set to True to automatically search for conda/mamba
AUTO_DETECT_CONDA = True
AUTO_DETECT_MAMBA = True

# Option 2: Specify custom paths (when auto-detection fails)
# Uncomment and modify the paths below if needed

# Conda paths to check (in order of preference)
CONDA_PATHS = [
    # Relative to user profile
    "~/Anaconda3/Scripts/conda.exe",
    "~/Miniconda3/Scripts/conda.exe",
    "~/anaconda3/Scripts/conda.exe",
    "~/miniconda3/Scripts/conda.exe",
    
    # System-wide installations
    "C:/ProgramData/Anaconda3/Scripts/conda.exe",
    "C:/ProgramData/Miniconda3/Scripts/conda.exe",
    
    # Custom paths (add your own here)
    # "D:/MyAnaconda/Scripts/conda.exe",
    # "C:/CustomPath/conda.exe",
]

# Mamba paths to check (in order of preference)
MAMBA_PATHS = [
    # Common mamba locations
    "~/.local/share/mamba/condabin/mamba.bat",
    "~/Anaconda3/Scripts/mamba.exe",
    "~/Miniconda3/Scripts/mamba.exe",
    "~/anaconda3/Scripts/mamba.exe",
    "~/miniconda3/Scripts/mamba.exe",
    
    # System-wide mamba installations
    "C:/ProgramData/Anaconda3/Scripts/mamba.exe",
    "C:/ProgramData/Miniconda3/Scripts/mamba.exe",
    
    # Custom paths (add your own here)
    # "D:/MyMamba/Scripts/mamba.exe",
    # "C:/CustomPath/mamba.exe",
]

# Package Manager Preferences
# ============================

# Preferred package manager order
# 1 = highest priority, 3 = lowest priority
PACKAGE_MANAGER_PRIORITY = {
    "mamba": 1,    # Fastest
    "conda": 2,    # Fast, good compatibility
    "pip": 3       # Fallback
}

# Conda/Mamba channels (in order of preference)
CONDA_CHANNELS = [
    "conda-forge",
    "defaults"
]

# Installation Options
# ====================

# Packages to install via conda/mamba (faster)
CONDA_PACKAGES = [
    "numpy",
    "pandas", 
    "requests",
    "pyyaml",
    "tqdm",
    "colorama",
    "matplotlib",
    "scipy"
]

# Packages that must be installed via pip (specialized)
PIP_ONLY_PACKAGES = [
    "pymilvus",
    "mcp",
    "fastmcp", 
    "sentence-transformers",
    "torch"
]

# Advanced Settings
# =================

# Maximum time to wait for package installation (seconds)
INSTALL_TIMEOUT = 300

# Enable verbose output
VERBOSE = True

# Create backup of environment before installation
CREATE_BACKUP = False

# User Instructions
# =================
"""
HOW TO CONFIGURE:

1. AUTO-DETECTION (Recommended):
   - Keep AUTO_DETECT_CONDA = True
   - Keep AUTO_DETECT_MAMBA = True
   - The script will automatically find your installations

2. CUSTOM PATHS:
   - Set AUTO_DETECT_* = False for manual configuration
   - Add your paths to CONDA_PATHS or MAMBA_PATHS lists
   - Use forward slashes (/) or double backslashes (\\\\)
   - Use ~ for user home directory

3. EXAMPLES:
   # Windows with custom Anaconda location
   CONDA_PATHS = ["D:/Anaconda3/Scripts/conda.exe"]
   
   # Linux/Mac with conda in custom location  
   CONDA_PATHS = ["~/my-conda/bin/conda"]
   
   # Multiple possible locations
   CONDA_PATHS = [
       "~/Anaconda3/Scripts/conda.exe",
       "C:/ProgramData/Anaconda3/Scripts/conda.exe",
       "D:/MyCondaInstall/Scripts/conda.exe"
   ]

4. TROUBLESHOOTING:
   - If auto-detection fails, set AUTO_DETECT_* = False
   - Add your specific paths to the *_PATHS lists
   - Run the script with VERBOSE = True for detailed logs
"""
