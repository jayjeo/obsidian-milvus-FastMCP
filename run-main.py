#!/usr/bin/env python3
"""
Obsidian-Milvus-FastMCP Main Program Launcher
Ensures proper working directory and imports for cross-platform compatibility
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher that ensures proper working directory"""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("================================================================")
    print("         Obsidian-Milvus-FastMCP Main Program")
    print("================================================================")
    print()
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Check if main.py exists
    main_py_path = script_dir / "main.py"
    if not main_py_path.exists():
        print("ERROR: main.py not found in current directory")
        print("Please make sure you're running this from the project folder")
        print()
        input("Press any key to continue...")
        sys.exit(1)
    
    print("Starting main.py...")
    print()
    
    try:
        # Run main.py with proper error handling
        result = subprocess.run([sys.executable, "main.py"], 
                              cwd=script_dir,
                              check=False)
        
        if result.returncode != 0:
            print()
            print(f"ERROR: main.py exited with error code {result.returncode}")
            print()
    
    except KeyboardInterrupt:
        print()
        print("Program interrupted by user")
        print()
    except Exception as e:
        print()
        print(f"ERROR: Failed to run main.py: {e}")
        print()
    
    print()
    print("Program finished.")
    input("Press any key to continue...")

if __name__ == "__main__":
    main()
