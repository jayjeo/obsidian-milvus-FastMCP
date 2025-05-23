"""
Path Detection Utility for Conda/Mamba
======================================

This script helps you find and configure conda/mamba paths for your system.
Run this to automatically detect your installations and generate config.py entries.
"""

import os
import sys
import shutil
from pathlib import Path


def find_executable(name):
    """Find executable using which/where command"""
    path = shutil.which(name)
    return path if path and os.path.exists(path) else None


def check_common_paths():
    """Check common installation paths for conda and mamba"""
    
    results = {
        'conda': [],
        'mamba': []
    }
    
    print("🔍 Scanning for Conda/Mamba installations...\n")
    
    # Common conda locations
    conda_locations = [
        # User installations
        Path.home() / "Anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "Miniconda3" / "Scripts" / "conda.exe", 
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "anaconda3" / "bin" / "conda",  # Linux/Mac
        Path.home() / "miniconda3" / "bin" / "conda",  # Linux/Mac
        
        # System-wide installations
        Path("C:/ProgramData/Anaconda3/Scripts/conda.exe"),
        Path("C:/ProgramData/Miniconda3/Scripts/conda.exe"),
        Path("/opt/anaconda3/bin/conda"),  # Linux/Mac
        Path("/opt/miniconda3/bin/conda"),  # Linux/Mac
        Path("/usr/local/bin/conda"),      # Linux/Mac
    ]
    
    # Common mamba locations  
    mamba_locations = [
        # User installations
        Path.home() / ".local" / "share" / "mamba" / "condabin" / "mamba.bat",
        Path.home() / ".local" / "share" / "mamba" / "condabin" / "mamba",
        Path.home() / "Anaconda3" / "Scripts" / "mamba.exe",
        Path.home() / "Miniconda3" / "Scripts" / "mamba.exe",
        Path.home() / "anaconda3" / "Scripts" / "mamba.exe", 
        Path.home() / "miniconda3" / "Scripts" / "mamba.exe",
        Path.home() / "anaconda3" / "bin" / "mamba",  # Linux/Mac
        Path.home() / "miniconda3" / "bin" / "mamba",  # Linux/Mac
        
        # System-wide installations
        Path("C:/ProgramData/Anaconda3/Scripts/mamba.exe"),
        Path("C:/ProgramData/Miniconda3/Scripts/mamba.exe"),
        Path("/opt/anaconda3/bin/mamba"),  # Linux/Mac
        Path("/opt/miniconda3/bin/mamba"),  # Linux/Mac
        Path("/usr/local/bin/mamba"),      # Linux/Mac
    ]
    
    # Check conda
    print("📦 Checking for Conda...")
    
    # First try PATH
    conda_in_path = find_executable("conda")
    if conda_in_path:
        results['conda'].append(("PATH", conda_in_path, check_executable(conda_in_path)))
        print(f"  ✓ Found in PATH: {conda_in_path}")
    
    # Then check common locations
    for path in conda_locations:
        if path.exists():
            working = check_executable(str(path))
            results['conda'].append(("Direct", str(path), working))
            status = "✓ Working" if working else "❌ Not working"
            print(f"  {status}: {path}")
    
    print(f"\n⚡ Checking for Mamba...")
    
    # First try PATH
    mamba_in_path = find_executable("mamba")
    if mamba_in_path:
        results['mamba'].append(("PATH", mamba_in_path, check_executable(mamba_in_path)))
        print(f"  ✓ Found in PATH: {mamba_in_path}")
    
    # Then check common locations
    for path in mamba_locations:
        if path.exists():
            working = check_executable(str(path))
            results['mamba'].append(("Direct", str(path), working))
            status = "✓ Working" if working else "❌ Not working"
            print(f"  {status}: {path}")
    
    return results


def check_executable(path):
    """Test if executable works by running --version"""
    try:
        import subprocess
        result = subprocess.run([path, "--version"], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except:
        return False


def generate_config_snippet(results):
    """Generate config.py snippet based on found paths"""
    
    print("\n" + "="*60)
    print("📝 CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    # Get working paths
    working_conda = [item for item in results['conda'] if item[2]]
    working_mamba = [item for item in results['mamba'] if item[2]]
    
    if not working_conda and not working_mamba:
        print("❌ No working conda or mamba installations found!")
        print("\n💡 Recommendations:")
        print("   1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        print("   2. Install Mamba: conda install mamba -c conda-forge")
        return
    
    print("\n🎯 Recommended config.py settings:\n")
    
    # Generate config snippet file
    config_content = []
    config_content.append("# Generated configuration snippet")
    config_content.append("# Copy these lines to your config.py")
    config_content.append("")
    
    if working_conda or working_mamba:
        print("# Automatic detection should work for your system")
        config_content.append("AUTO_DETECT_CONDA = True") 
        config_content.append("AUTO_DETECT_MAMBA = True")
        print("AUTO_DETECT_CONDA = True")
        print("AUTO_DETECT_MAMBA = True")
        print()
        config_content.append("")
    
    if working_conda:
        print("# If auto-detection fails, use these conda paths:")
        print("CONDA_PATHS = [")
        config_content.append("CONDA_PATHS = [")
        for method, path, working in working_conda:
            # Convert to relative path if possible
            relative_path = make_relative_path(path)
            print(f'    "{relative_path}",  # {method} - Working')
            config_content.append(f'    "{relative_path}",  # {method} - Working')
        print("]")
        config_content.append("]")
        print()
        config_content.append("")
    
    if working_mamba:
        print("# If auto-detection fails, use these mamba paths:")
        print("MAMBA_PATHS = [")
        config_content.append("MAMBA_PATHS = [")
        for method, path, working in working_mamba:
            relative_path = make_relative_path(path)
            print(f'    "{relative_path}",  # {method} - Working')
            config_content.append(f'    "{relative_path}",  # {method} - Working')
        print("]")
        config_content.append("]")
        print()
        config_content.append("")
    
    # Priority recommendations
    print("# Recommended priority:")
    if working_mamba:
        print('PACKAGE_MANAGER_PRIORITY = {')
        print('    "mamba": 1,    # Fastest - you have this!')
        print('    "conda": 2,')
        print('    "pip": 3')
        print('}')
        config_content.extend([
            'PACKAGE_MANAGER_PRIORITY = {',
            '    "mamba": 1,    # Fastest - you have this!',
            '    "conda": 2,',
            '    "pip": 3',
            '}'
        ])
    elif working_conda:
        print('PACKAGE_MANAGER_PRIORITY = {')
        print('    "conda": 1,    # You have this')
        print('    "mamba": 2,    # Install with: conda install mamba -c conda-forge')
        print('    "pip": 3')
        print('}')
        config_content.extend([
            'PACKAGE_MANAGER_PRIORITY = {',
            '    "conda": 1,    # You have this',
            '    "mamba": 2,    # Install with: conda install mamba -c conda-forge',
            '    "pip": 3',
            '}'
        ])
    
    # Save config snippet to file
    try:
        with open('config-snippet.txt', 'w') as f:
            f.write('\n'.join(config_content))
        print("\n📄 Configuration snippet saved to: config-snippet.txt")
    except Exception as e:
        print(f"\n⚠️  Could not save config snippet: {e}")


def make_relative_path(path):
    """Convert absolute path to relative path using ~ for home directory"""
    path_obj = Path(path)
    home = Path.home()
    
    try:
        # Try to make it relative to home directory
        relative = path_obj.relative_to(home)
        return f"~/{relative.as_posix()}"
    except ValueError:
        # If not under home directory, return as-is with forward slashes
        return path.replace("\\", "/")


def main():
    print("="*60)
    print("🔧 Conda/Mamba Path Detection Utility")
    print("="*60)
    print()
    
    # Detect installations
    results = check_common_paths()
    
    # Generate recommendations
    generate_config_snippet(results)
    
    print("\n" + "="*60)
    print("✅ PATH DETECTION COMPLETE")
    print("="*60)
    print()
    print("📋 Next steps:")
    print("   1. Copy the recommended settings above to your config.py")
    print("   2. Adjust paths if needed for your specific setup")
    print("   3. Run smart-install-config.bat to test the configuration")
    print()


if __name__ == "__main__":
    main()
