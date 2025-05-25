#!/usr/bin/env python3
"""
MCP Server Startup Diagnostics and Timeout Handler
Diagnoses MCP server startup issues and implements timeout mechanisms
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path

# Configure logging
log_file = Path(__file__).parent / 'mcp_startup_debug.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimeoutHandler:
    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.process = None
        
    def start_timer(self):
        self.start_time = time.time()
        
    def check_timeout(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout_seconds
        
    def kill_process(self):
        if self.process and self.process.poll() is None:
            logger.warning(f"Killing process due to timeout ({self.timeout_seconds}s)")
            self.process.terminate()
            time.sleep(2)
            if self.process.poll() is None:
                self.process.kill()

def check_environment():
    """Check Python environment"""
    logger.info('Checking Python environment...')
    try:
        import markdown
        logger.info(f'✅ markdown module available: {markdown.__version__}')
        
        import pymilvus
        logger.info(f'✅ pymilvus module available')
        
        import mcp
        logger.info(f'✅ mcp module available')
        
        import fastmcp  
        logger.info(f'✅ fastmcp module available')
        
        return True
    except ImportError as e:
        logger.error(f'❌ Module import error: {e}')
        return False
    except Exception as e:
        logger.error(f'❌ Unexpected error: {e}')
        return False

def check_milvus_connection():
    """Check Milvus connection"""
    logger.info('Testing Milvus connection...')
    try:
        from pymilvus import connections
        
        # Test connection with timeout
        connections.connect(
            alias="default",
            host="localhost", 
            port="19530",
            timeout=10
        )
        
        logger.info('✅ Milvus connection successful')
        connections.disconnect("default")
        return True
        
    except Exception as e:
        logger.error(f'❌ Milvus connection failed: {e}')
        return False

def test_mcp_server_import():
    """Test if MCP server script can be imported"""
    logger.info('Testing MCP server script import...')
    try:
        script_dir = Path(__file__).parent
        mcp_server_path = script_dir / 'mcp_server.py'
        
        if not mcp_server_path.exists():
            logger.error(f'❌ MCP server script not found: {mcp_server_path}')
            return False
            
        # Test basic syntax by compiling
        with open(mcp_server_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        compile(source_code, str(mcp_server_path), 'exec')
        logger.info('✅ MCP server script syntax OK')
        return True
        
    except SyntaxError as e:
        logger.error(f'❌ MCP server script syntax error: {e}')
        return False
    except Exception as e:
        logger.error(f'❌ MCP server script error: {e}')
        return False

def start_mcp_server_with_timeout(timeout_seconds=60):
    """Start MCP server with timeout and monitoring"""
    logger.info(f'Starting MCP Server with {timeout_seconds}s timeout...')
    
    script_dir = Path(__file__).parent
    mcp_server_path = script_dir / 'mcp_server.py'
    
    if not mcp_server_path.exists():
        logger.error(f'❌ MCP server script not found: {mcp_server_path}')
        return False
    
    timeout_handler = TimeoutHandler(timeout_seconds)
    
    try:
        # Start MCP server process
        logger.info(f'Executing: {sys.executable} {mcp_server_path}')
        
        process = subprocess.Popen(
            [sys.executable, str(mcp_server_path)],
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        timeout_handler.process = process
        timeout_handler.start_timer()
        
        # Monitor process output with timeout
        output_lines = []
        error_lines = []
        
        def read_stdout():
            try:
                for line in process.stdout:
                    output_lines.append(line.strip())
                    logger.info(f'MCP-STDOUT: {line.strip()}')
                    if len(output_lines) > 100:  # Prevent memory buildup
                        output_lines.pop(0)
            except Exception as e:
                logger.error(f'Error reading stdout: {e}')
        
        def read_stderr():
            try:
                for line in process.stderr:
                    error_lines.append(line.strip())
                    logger.error(f'MCP-STDERR: {line.strip()}')
                    if len(error_lines) > 100:  # Prevent memory buildup
                        error_lines.pop(0)
            except Exception as e:
                logger.error(f'Error reading stderr: {e}')
        
        # Start output monitoring threads
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process with timeout checking
        start_time = time.time()
        while process.poll() is None:
            if timeout_handler.check_timeout():
                logger.warning(f'⏰ MCP Server startup timeout ({timeout_seconds}s)')
                timeout_handler.kill_process()
                return False
                
            # Check if we got any meaningful output indicating startup
            if output_lines:
                for line in output_lines[-5:]:  # Check last 5 lines
                    if any(keyword in line.lower() for keyword in 
                          ['server started', 'listening', 'ready', 'started successfully']):
                        logger.info('✅ MCP Server appears to have started successfully')
                        # Give it a bit more time to fully initialize
                        time.sleep(5)
                        if process.poll() is None:
                            logger.info('✅ MCP Server is running')
                            return True
                        else:
                            logger.error('❌ MCP Server exited unexpectedly')
                            return False
            
            time.sleep(1)  # Check every second
        
        # Process exited
        return_code = process.returncode
        logger.info(f'MCP Server exited with code: {return_code}')
        
        if return_code == 0:
            logger.info('✅ MCP Server completed successfully')
            return True
        else:
            logger.error(f'❌ MCP Server failed with code: {return_code}')
            
            # Show last few lines of output/error
            if output_lines:
                logger.info('Last stdout lines:')
                for line in output_lines[-5:]:
                    logger.info(f'  {line}')
                    
            if error_lines:
                logger.error('Last stderr lines:')
                for line in error_lines[-5:]:
                    logger.error(f'  {line}')
                    
            return False
            
    except Exception as e:
        logger.error(f'❌ Error starting MCP server: {e}')
        timeout_handler.kill_process()
        return False

def initialize_with_diagnostics():
    """Initialize system with comprehensive diagnostics"""
    logger.info('=== COMPREHENSIVE MCP STARTUP DIAGNOSTICS ===')
    
    # Step 1: Environment check
    logger.info('Step 1: Environment validation...')
    if not check_environment():
        logger.error('❌ Environment check failed')
        return False
    
    # Step 2: Milvus connection test
    logger.info('Step 2: Milvus connection test...')
    if not check_milvus_connection():
        logger.error('❌ Milvus connection test failed')
        return False
    
    # Step 3: MCP server script validation
    logger.info('Step 3: MCP server script validation...')
    if not test_mcp_server_import():
        logger.error('❌ MCP server script validation failed')
        return False
    
    # Step 4: Initialize components (quick test)
    logger.info('Step 4: Quick component initialization test...')
    try:
        from milvus_manager import MilvusManager
        from obsidian_processor import ObsidianProcessor
        import config
        
        logger.info('✅ Component imports successful')
        
        # Quick Milvus manager test
        milvus_manager = MilvusManager()
        logger.info('✅ MilvusManager created')
        
        # Don't create ObsidianProcessor as it loads heavy models
        logger.info('✅ Core initialization successful')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Component initialization failed: {e}')
        return False

def main():
    """Main diagnostics and startup function"""
    logger.info('Starting MCP Server diagnostics and startup...')
    
    # Run diagnostics first
    if not initialize_with_diagnostics():
        logger.error('❌ Diagnostics failed - aborting startup')
        return False
    
    logger.info('✅ All diagnostics passed')
    
    # Start MCP server with timeout
    logger.info('Starting MCP Server with timeout protection...')
    success = start_mcp_server_with_timeout(timeout_seconds=60)
    
    if success:
        logger.info('✅ MCP Server startup completed')
    else:
        logger.error('❌ MCP Server startup failed or timed out')
        
        # Suggest alternative startup methods
        logger.info('')
        logger.info('=== ALTERNATIVE STARTUP SUGGESTIONS ===')
        logger.info('Try these alternative methods:')
        logger.info('1. Direct MCP server: python mcp_server.py')
        logger.info('2. Test mode: python mcp_server.py --test')
        logger.info('3. Check Claude Desktop config')
        logger.info('4. Run main.py instead for interactive mode')
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info('⏹️ Interrupted by user')
        sys.exit(1)
    except Exception as e:
        logger.error(f'💥 Unexpected error: {e}')
        import traceback
        logger.error(f'Stack trace: {traceback.format_exc()}')
        sys.exit(1)
