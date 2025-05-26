#!/usr/bin/env python3
"""
CUDA Toolkit Installation Checker
This script checks whether CUDA Toolkit is installed and provides detailed information.
"""

import os
import subprocess
import sys
import platform
from pathlib import Path

def run_command(command):
    """Run a command and return its output, return None if command fails"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None

def check_nvcc():
    """Check if nvcc (NVIDIA CUDA Compiler) is available"""
    print("🔍 Checking NVCC (CUDA Compiler)...")
    
    nvcc_version = run_command("nvcc --version")
    if nvcc_version:
        print("✅ NVCC found!")
        # Extract version info
        for line in nvcc_version.split('\n'):
            if 'release' in line.lower():
                print(f"   Version: {line.strip()}")
                break
        return True
    else:
        print("❌ NVCC not found in PATH")
        return False

def check_nvidia_smi():
    """Check if nvidia-smi is available"""
    print("\n🔍 Checking NVIDIA-SMI...")
    
    nvidia_smi = run_command("nvidia-smi --version")
    if nvidia_smi:
        print("✅ NVIDIA-SMI found!")
        for line in nvidia_smi.split('\n'):
            if 'NVIDIA-SMI' in line:
                print(f"   {line.strip()}")
                break
        return True
    else:
        print("❌ NVIDIA-SMI not found")
        return False

def check_cuda_runtime():
    """Check CUDA runtime version"""
    print("\n🔍 Checking CUDA Runtime...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        major = cuda_version // 1000
        minor = (cuda_version % 1000) // 10
        print(f"✅ CUDA Runtime Version: {major}.{minor}")
        return True
    except ImportError:
        # Try alternative method
        nvidia_smi_output = run_command("nvidia-smi")
        if nvidia_smi_output:
            for line in nvidia_smi_output.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"✅ CUDA Runtime Version: {cuda_version}")
                    return True
        print("❌ Could not determine CUDA Runtime version")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA Runtime: {e}")
        return False

def check_cuda_paths():
    """Check common CUDA installation paths"""
    print("\n🔍 Checking CUDA Installation Paths...")
    
    common_paths = []
    system = platform.system().lower()
    
    if system == "windows":
        common_paths = [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            "C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"
        ]
    elif system == "linux":
        common_paths = [
            "/usr/local/cuda",
            "/usr/cuda",
            "/opt/cuda"
        ]
    elif system == "darwin":  # macOS
        common_paths = [
            "/usr/local/cuda",
            "/Developer/NVIDIA/CUDA"
        ]
    
    found_paths = []
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Found CUDA at: {path}")
            found_paths.append(path)
            # List versions if multiple exist
            try:
                versions = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('v')]
                if versions:
                    print(f"   Versions: {', '.join(sorted(versions))}")
            except PermissionError:
                print("   (Cannot list versions - permission denied)")
    
    if not found_paths:
        print("❌ No CUDA installations found in common paths")
        return False
    
    return True

def check_environment_variables():
    """Check CUDA-related environment variables"""
    print("\n🔍 Checking Environment Variables...")
    
    cuda_vars = [
        'CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT',
        'CUDA_BIN_PATH', 'CUDA_LIB_PATH', 'CUDA_INC_PATH'
    ]
    
    found_vars = {}
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            found_vars[var] = value
            print(f"✅ {var}: {value}")
    
    if not found_vars:
        print("❌ No CUDA environment variables found")
        return False
    
    return True

def check_path_variable():
    """Check if CUDA binaries are in PATH"""
    print("\n🔍 Checking PATH for CUDA binaries...")
    
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    cuda_in_path = []
    
    for path_dir in path_dirs:
        if 'cuda' in path_dir.lower():
            cuda_in_path.append(path_dir)
            print(f"✅ CUDA path in PATH: {path_dir}")
    
    if not cuda_in_path:
        print("❌ No CUDA paths found in PATH variable")
        return False
    
    return True

def check_python_cuda_libraries():
    """Check if Python CUDA libraries are installed"""
    print("\n🔍 Checking Python CUDA Libraries...")
    
    libraries = {
        'pycuda': 'PyCUDA',
        'cupy': 'CuPy', 
        'numba': 'Numba (with CUDA)',
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow'
    }
    
    found_libs = []
    for lib_name, display_name in libraries.items():
        try:
            if lib_name == 'numba':
                import numba.cuda
                if numba.cuda.is_available():
                    found_libs.append(display_name)
                    print(f"✅ {display_name} available")
            elif lib_name == 'torch':
                import torch
                if torch.cuda.is_available():
                    found_libs.append(f"{display_name} (CUDA {torch.version.cuda})")
                    print(f"✅ {display_name} with CUDA {torch.version.cuda}")
            elif lib_name == 'tensorflow':
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    found_libs.append(display_name)
                    print(f"✅ {display_name} with GPU support")
            else:
                __import__(lib_name)
                found_libs.append(display_name)
                print(f"✅ {display_name} installed")
        except ImportError:
            print(f"❌ {display_name} not installed")
        except Exception as e:
            print(f"⚠️  {display_name} installed but error checking CUDA: {e}")
    
    return len(found_libs) > 0

def main():
    """Main function to run all checks"""
    print("=" * 60)
    print("🚀 CUDA TOOLKIT INSTALLATION CHECKER")
    print("=" * 60)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    
    checks = [
        ("NVCC Compiler", check_nvcc),
        ("NVIDIA-SMI", check_nvidia_smi),
        ("CUDA Runtime", check_cuda_runtime),
        ("Installation Paths", check_cuda_paths),
        ("Environment Variables", check_environment_variables),
        ("PATH Variable", check_path_variable),
        ("Python Libraries", check_python_cuda_libraries)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ Error during {check_name} check: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    passed_checks = sum(results.values())
    total_checks = len(results)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<20}: {status}")
    
    print("-" * 60)
    print(f"Overall Score: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks >= 3:  # At least basic CUDA components
        print("🎉 CUDA Toolkit appears to be installed!")
    elif passed_checks >= 1:
        print("⚠️  CUDA Toolkit may be partially installed or misconfigured")
    else:
        print("❌ CUDA Toolkit does not appear to be installed")
    
    print("\n💡 Tips:")
    if not results.get("NVCC Compiler", False):
        print("   - Install CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit")
    if not results.get("Environment Variables", False):
        print("   - Set CUDA_PATH environment variable to your CUDA installation")
    if not results.get("PATH Variable", False):
        print("   - Add CUDA bin directory to your PATH variable")

if __name__ == "__main__":
    main()