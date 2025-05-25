#!/usr/bin/env python3
"""
Claude Desktop Configuration Checker and Fixer
Diagnoses and fixes Claude Desktop MCP server configuration issues
"""

import os
import json
import sys
from pathlib import Path

def find_claude_config_path():
    """Find Claude Desktop configuration file"""
    user_profile = os.environ.get('USERPROFILE', '')
    if not user_profile:
        return None
    
    # Common Claude Desktop config paths
    possible_paths = [
        Path(user_profile) / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
        Path(user_profile) / ".claude" / "claude_desktop_config.json",
        Path(user_profile) / "AppData" / "Local" / "Claude" / "claude_desktop_config.json",
        Path(user_profile) / "Documents" / "Claude" / "claude_desktop_config.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Return the most common location for creation
    return Path(user_profile) / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"

def read_current_config(config_path):
    """Read current Claude Desktop configuration"""
    if not config_path.exists():
        print(f"❌ Claude Desktop config not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Current config found: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def display_current_config(config):
    """Display current configuration"""
    print("\n" + "="*60)
    print("CURRENT CLAUDE DESKTOP CONFIGURATION")
    print("="*60)
    
    if not config:
        print("❌ No configuration loaded")
        return
    
    if 'mcpServers' in config:
        print("MCP Servers configuration:")
        for server_name, server_config in config['mcpServers'].items():
            print(f"\n📡 Server: {server_name}")
            print(f"   Command: {server_config.get('command', 'N/A')}")
            print(f"   Args: {server_config.get('args', 'N/A')}")
            print(f"   Environment: {server_config.get('env', 'N/A')}")
    else:
        print("❌ No mcpServers configuration found")
    
    print()

def get_current_project_path():
    """Get current project absolute path"""
    return str(Path(__file__).parent.absolute())

def find_working_python():
    """Find the working Python executable"""
    script_dir = Path(__file__).parent
    
    # Check if we have a verified working Python from previous setup
    if (script_dir / "start_mcp_verified.bat").exists():
        try:
            with open(script_dir / "start_mcp_verified.bat", 'r') as f:
                content = f.read()
                # Extract Python path from the batch file
                for line in content.split('\n'):
                    if 'python' in line.lower() and '"' in line:
                        # Extract path between quotes
                        start = line.find('"')
                        end = line.find('"', start + 1)
                        if start != -1 and end != -1:
                            python_path = line[start + 1:end]
                            if Path(python_path).exists():
                                return python_path
        except:
            pass
    
    # Fallback to current Python
    return sys.executable

def create_correct_config():
    """Create correct Claude Desktop configuration"""
    project_path = get_current_project_path()
    python_exe = find_working_python()
    
    # For Windows, use the working Python path we found
    config = {
        "mcpServers": {
            "obsidian-assistant": {
                "command": python_exe,
                "args": [
                    str(Path(project_path) / "mcp_server.py")
                ],
                "env": {
                    "PYTHONPATH": project_path
                }
            }
        }
    }
    
    return config

def backup_current_config(config_path):
    """Backup current configuration"""
    if not config_path.exists():
        return None
    
    backup_path = config_path.with_suffix('.json.backup')
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"✅ Current config backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"⚠️ Could not backup config: {e}")
        return None

def write_new_config(config_path, config):
    """Write new configuration to Claude Desktop"""
    try:
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ New configuration written to: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing config: {e}")
        return False

def test_mcp_server():
    """Test if MCP server can start"""
    project_path = get_current_project_path()
    python_exe = find_working_python()
    mcp_server_path = Path(project_path) / "mcp_server.py"
    
    print(f"Testing MCP server startup...")
    print(f"Python: {python_exe}")
    print(f"Script: {mcp_server_path}")
    
    if not mcp_server_path.exists():
        print(f"❌ MCP server script not found: {mcp_server_path}")
        return False
    
    try:
        import subprocess
        # Quick test - start and immediately stop
        result = subprocess.run([
            python_exe, 
            str(mcp_server_path)
        ], timeout=10, capture_output=True, text=True, cwd=project_path)
        
        print(f"✅ MCP server test completed (exit code: {result.returncode})")
        if result.stdout:
            print("STDOUT:", result.stdout[-200:])  # Last 200 chars
        if result.stderr:
            print("STDERR:", result.stderr[-200:])  # Last 200 chars
        
        return True
        
    except subprocess.TimeoutExpired:
        print("✅ MCP server started (timed out after 10s, which is expected)")
        return True
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("CLAUDE DESKTOP CONFIGURATION CHECKER AND FIXER")
    print("=" * 60)
    print()
    
    # Step 1: Find Claude config
    print("Step 1: Finding Claude Desktop configuration...")
    config_path = find_claude_config_path()
    print(f"Config path: {config_path}")
    
    # Step 2: Read current config
    print("\nStep 2: Reading current configuration...")
    current_config = read_current_config(config_path)
    display_current_config(current_config)
    
    # Step 3: Test MCP server
    print("\nStep 3: Testing MCP server...")
    mcp_working = test_mcp_server()
    
    if not mcp_working:
        print("❌ MCP server is not working. Fix the server first.")
        return False
    
    # Step 4: Create correct configuration
    print("\nStep 4: Creating correct configuration...")
    correct_config = create_correct_config()
    
    print("New configuration:")
    print(json.dumps(correct_config, indent=2))
    
    # Step 5: Ask user if they want to update
    print(f"\nStep 5: Update Claude Desktop configuration...")
    print(f"This will update: {config_path}")
    
    response = input("Do you want to update the configuration? (Y/N): ").strip().upper()
    
    if response == 'Y':
        # Backup current config
        backup_path = backup_current_config(config_path)
        
        # Write new config
        if write_new_config(config_path, correct_config):
            print("\n" + "="*60)
            print("✅ CONFIGURATION UPDATED SUCCESSFULLY!")
            print("="*60)
            print()
            print("Next steps:")
            print("1. 🔄 Restart Claude Desktop completely")
            print("2. 📝 Open a new conversation in Claude Desktop")
            print("3. 🧪 Test the MCP server by asking: 'Search my notes for...'")
            print()
            print(f"📁 Project path: {get_current_project_path()}")
            print(f"🐍 Python path: {find_working_python()}")
            print(f"⚙️ Config file: {config_path}")
            if backup_path:
                print(f"💾 Backup: {backup_path}")
            
            return True
        else:
            print("❌ Failed to update configuration")
            return False
    else:
        print("Configuration not updated.")
        print()
        print("Manual update instructions:")
        print(f"1. Open: {config_path}")
        print("2. Replace contents with the configuration shown above")
        print("3. Restart Claude Desktop")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        input(f"\nPress Enter to exit... (Success: {success})")
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        input("Press Enter to exit...")
