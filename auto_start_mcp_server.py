#!/usr/bin/env python3
"""
Auto Start MCP Server - Extracted from main.py option 1
This script automatically starts the MCP server for Claude Desktop integration.
Designed to be called by VBS script for Windows startup automation.
"""

import os
import sys
import threading
import time
import logging
from pathlib import Path

# Configure logging for startup automation
log_file = Path(__file__).parent / "auto_startup_mcp.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the Obsidian-Milvus-MCP system"""
    try:
        logger.info("Obsidian-Milvus-MCP system initializing...")
        
        # Import required modules
        from milvus_manager import MilvusManager
        from obsidian_processor import ObsidianProcessor
        from watcher import start_watcher
        import config
        
        # Initialize Milvus connection and collection
        logger.info("Connecting to Milvus...")
        milvus_manager = MilvusManager()
        
        # Initialize Obsidian processor
        logger.info("Initializing Obsidian processor...")
        processor = ObsidianProcessor(milvus_manager)
        
        # Check for existing data
        try:
            results = milvus_manager.query("id >= 0", limit=1)
            if not results:
                logger.info("No existing data found. MCP server ready for indexing.")
            else:
                entity_count = milvus_manager.count_entities()
                logger.info(f"Found existing data: {entity_count} documents indexed.")
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}")
        
        return processor
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def start_file_watcher(processor):
    """Start the file change detection watcher"""
    try:
        from watcher import start_watcher
        import config
        
        logger.info("Starting file watcher...")
        watcher_thread = threading.Thread(target=start_watcher, args=(processor,))
        watcher_thread.daemon = True
        watcher_thread.start()
        
        logger.info("File watcher started successfully")
        logger.info(f"Monitoring directory: {config.OBSIDIAN_VAULT_PATH}")
        
        return watcher_thread
        
    except Exception as e:
        logger.error(f"Failed to start file watcher: {e}")
        return None

def start_mcp_server():
    """Start the MCP server for Claude Desktop - Extracted from main.py option 1"""
    try:
        import config
        
        logger.info("Starting MCP Server for Claude Desktop...")
        logger.info(f"Server name: {config.FASTMCP_SERVER_NAME}")
        logger.info(f"Transport: {config.FASTMCP_TRANSPORT}")
        
        # Get the path to mcp_server.py using relative path from config
        mcp_server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        
        if not os.path.exists(mcp_server_path):
            raise FileNotFoundError(f"MCP server script not found: {mcp_server_path}")
        
        logger.info(f"Executing MCP server: {mcp_server_path}")
        
        # Execute the MCP server (this will block and run the server)
        os.system(f'python "{mcp_server_path}"')
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise

def main():
    """Main function that replicates main.py option 1 functionality"""
    try:
        logger.info("=" * 60)
        logger.info("Auto Start MCP Server - Starting...")
        logger.info("=" * 60)
        
        # Initialize the system
        processor = initialize_system()
        
        # Start file watcher
        watcher_thread = start_file_watcher(processor)
        
        if watcher_thread is None:
            logger.warning("File watcher failed to start, but continuing with MCP server...")
        
        # Small delay to ensure everything is initialized
        time.sleep(2)
        
        # Start MCP server (this will block)
        logger.info("Ready to start MCP server...")
        start_mcp_server()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        logger.error("Exiting with error code 1")
        sys.exit(1)

if __name__ == "__main__":
    main()
