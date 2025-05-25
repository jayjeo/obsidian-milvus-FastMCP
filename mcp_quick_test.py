#!/usr/bin/env python3
"""
Quick MCP Server Test - Bypass Heavy Initialization
Tests if MCP server can start without heavy model loading
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Configure logging
log_file = Path(__file__).parent / 'mcp_quick_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_mcp_server_direct():
    """Test MCP server directly without initialization"""
    logger.info("Testing MCP server direct startup...")
    
    script_dir = Path(__file__).parent
    mcp_server_path = script_dir / 'mcp_server.py'
    
    if not mcp_server_path.exists():
        logger.error(f"MCP server script not found: {mcp_server_path}")
        return False
    
    try:
        # Start MCP server with short timeout
        logger.info("Starting MCP server with 30-second test timeout...")
        
        process = subprocess.Popen(
            [sys.executable, str(mcp_server_path)],
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Process started with PID: {process.pid}")
        
        # Wait for 30 seconds max
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                # Process finished
                stdout, stderr = process.communicate()
                logger.info(f"Process exited with code: {process.returncode}")
                
                if stdout:
                    logger.info("STDOUT:")
                    for line in stdout.split('\n')[-10:]:  # Last 10 lines
                        if line.strip():
                            logger.info(f"  {line}")
                
                if stderr:
                    logger.error("STDERR:")
                    for line in stderr.split('\n')[-10:]:  # Last 10 lines
                        if line.strip():
                            logger.error(f"  {line}")
                
                return process.returncode == 0
            
            time.sleep(1)
        
        # Timeout reached
        logger.warning("Test timeout reached - terminating process")
        process.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(3)
        if process.poll() is None:
            logger.warning("Force killing process")
            process.kill()
        
        return False
        
    except Exception as e:
        logger.error(f"Error testing MCP server: {e}")
        return False

def check_claude_desktop_config():
    """Check if Claude Desktop config exists and is valid"""
    logger.info("Checking Claude Desktop configuration...")
    
    # Common Claude Desktop config paths
    user_profile = os.environ.get('USERPROFILE', '')
    config_paths = []
    
    if user_profile:
        config_paths.extend([
            Path(user_profile) / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
            Path(user_profile) / ".claude" / "claude_desktop_config.json",
        ])
    
    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Found Claude Desktop config: {config_path}")
            try:
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                if 'mcpServers' in config_data:
                    logger.info("MCP servers configuration found")
                    for server_name, server_config in config_data['mcpServers'].items():
                        logger.info(f"  Server: {server_name}")
                        if 'command' in server_config:
                            logger.info(f"    Command: {server_config['command']}")
                        if 'args' in server_config:
                            logger.info(f"    Args: {server_config['args']}")
                    return True
                else:
                    logger.warning("No MCP servers configuration found in Claude Desktop config")
                    
            except Exception as e:
                logger.error(f"Error reading Claude Desktop config: {e}")
    
    logger.warning("No Claude Desktop configuration found")
    return False

def suggest_solutions():
    """Suggest solutions based on the analysis"""
    logger.info("")
    logger.info("=== SUGGESTED SOLUTIONS ===")
    
    logger.info("The MCP server appears to hang during startup. Here are solutions:")
    logger.info("")
    
    logger.info("1. QUICK FIX - Use timeout protection:")
    logger.info("   start_mcp_with_timeout.bat")
    logger.info("")
    
    logger.info("2. ALTERNATIVE - Direct startup:")
    logger.info("   python mcp_server.py")
    logger.info("   (May hang, but you can Ctrl+C to stop)")
    logger.info("")
    
    logger.info("3. INTERACTIVE MODE - Use main menu:")
    logger.info("   python main.py")
    logger.info("   (Choose option 1 from menu)")
    logger.info("")
    
    logger.info("4. LIGHTWEIGHT MODE - Skip heavy initialization:")
    logger.info("   Edit obsidian_processor.py to skip model loading")
    logger.info("")
    
    logger.info("5. CHECK CLAUDE DESKTOP - Ensure proper configuration:")
    logger.info("   Run setup.py option 5 to generate config")
    logger.info("")
    
    logger.info("6. DEBUG MODE - Check what's causing the hang:")
    logger.info("   python mcp_startup_diagnostics.py")

def main():
    logger.info("=== MCP SERVER QUICK TEST ===")
    logger.info("Testing MCP server startup to identify hanging issues...")
    logger.info("")
    
    # Test 1: Check Claude Desktop config
    logger.info("Test 1: Claude Desktop configuration")
    check_claude_desktop_config()
    logger.info("")
    
    # Test 2: Quick MCP server test
    logger.info("Test 2: Direct MCP server startup test")
    success = test_mcp_server_direct()
    
    if success:
        logger.info("✅ MCP server can start successfully")
    else:
        logger.error("❌ MCP server has startup issues")
    
    logger.info("")
    
    # Provide solutions
    suggest_solutions()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)
