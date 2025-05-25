#!/usr/bin/env python3
"""
Admin Rights Checker - Determines if administrator privileges are actually needed
"""

import os
import sys
from pathlib import Path

def check_admin_requirements():
    """Check if admin rights are actually needed for this project"""
    
    print("================================================================")
    print("Administrator Rights Requirement Analysis")
    print("================================================================")
    print()
    
    script_dir = Path(__file__).parent
    
    # Check if running as admin on Windows
    if os.name == 'nt':
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            print(f"Currently running as administrator: {is_admin}")
        except:
            print("Cannot determine admin status")
    
    print()
    
    # Check file permissions
    print("Checking file permissions...")
    print("-" * 40)
    
    # Test write access to project directory
    test_file = script_dir / "temp_permission_test.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Project directory: WRITE ACCESS OK")
        project_writable = True
    except Exception as e:
        print(f"❌ Project directory: NO WRITE ACCESS - {e}")
        project_writable = False
    
    # Check MilvusData directory
    milvus_data = script_dir / "MilvusData"
    try:
        milvus_data.mkdir(exist_ok=True)
        test_file = milvus_data / "temp_test.txt"
        test_file.write_text("test")
        test_file.unlink()
        print("✅ MilvusData directory: WRITE ACCESS OK")
        milvus_writable = True
    except Exception as e:
        print(f"❌ MilvusData directory: NO WRITE ACCESS - {e}")
        milvus_writable = False
    
    # Check volumes directory
    volumes_dir = script_dir / "volumes"
    try:
        volumes_dir.mkdir(exist_ok=True)
        test_file = volumes_dir / "temp_test.txt"
        test_file.write_text("test")
        test_file.unlink()
        print("✅ volumes directory: WRITE ACCESS OK")
        volumes_writable = True
    except Exception as e:
        print(f"❌ volumes directory: NO WRITE ACCESS - {e}")
        volumes_writable = False
    
    print()
    
    # Check Python package access
    print("Checking Python package access...")
    print("-" * 40)
    
    critical_packages = ['markdown', 'pymilvus', 'sentence_transformers', 'watchdog']
    packages_available = True
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}: AVAILABLE")
        except ImportError:
            print(f"❌ {package}: NOT AVAILABLE")
            packages_available = False
    
    print()
    
    # Check network access (for MCP server)
    print("Checking network capabilities...")
    print("-" * 40)
    
    try:
        import socket
        # Test binding to MCP server ports
        test_ports = [5680, 19530, 9091]
        for port in test_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                print(f"✅ Port {port}: BINDABLE")
            except Exception as e:
                print(f"⚠️  Port {port}: {e}")
    except Exception as e:
        print(f"❌ Network test failed: {e}")
    
    print()
    
    # Analysis and recommendations
    print("================================================================")
    print("Analysis and Recommendations")
    print("================================================================")
    print()
    
    admin_needed = False
    reasons = []
    
    if not project_writable:
        admin_needed = True
        reasons.append("Project directory requires write access")
    
    if not milvus_writable:
        admin_needed = True
        reasons.append("MilvusData directory requires write access")
    
    if not volumes_writable:
        admin_needed = True
        reasons.append("volumes directory requires write access")
    
    if not packages_available:
        reasons.append("Python packages not available (environment issue)")
    
    if admin_needed:
        print("❌ ADMINISTRATOR RIGHTS REQUIRED")
        print()
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
        print()
        print("Solutions:")
        print("  1. Run as administrator")
        print("  2. Change folder permissions")
        print("  3. Move project to a user-writable location")
    else:
        print("✅ ADMINISTRATOR RIGHTS NOT REQUIRED")
        print()
        print("You can run this program with normal user privileges.")
        
        if not packages_available:
            print()
            print("⚠️  However, Python environment issues detected:")
            for reason in reasons:
                print(f"  - {reason}")
            print()
            print("This is likely because:")
            print("  - Different Python environment when running as admin")
            print("  - Packages installed only for current user")
            print()
            print("Recommended solutions:")
            print("  1. Run WITHOUT administrator privileges")
            print("  2. Or install packages system-wide: pip install --upgrade [packages]")
    
    print()
    return not admin_needed and packages_available

if __name__ == "__main__":
    success = check_admin_requirements()
    if not success:
        input("Press Enter to continue...")
    sys.exit(0 if success else 1)
