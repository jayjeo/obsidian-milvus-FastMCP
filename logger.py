"""
Centralized logging module for Obsidian-Milvus-FastMCP project.
All application modules should use this logger for consistent logging.
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler
import sys
import config

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"obsidian_milvus_{current_date}.log")

# Get log level from config or use INFO as default
try:
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_str, logging.INFO)
except (ImportError, AttributeError):
    log_level = logging.INFO

# Configure root logger
def setup_logger():
    """Configure the root logger with console and file handlers"""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set the root logger level
    root_logger.setLevel(log_level)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Create console handler - MUST use stderr for MCP compatibility
    # MCP uses stdout for JSON-RPC communication, so we must use stderr for logging
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

# Initialize the root logger
root_logger = setup_logger()

def get_logger(name):
    """Get a logger with the specified name that writes to the centralized log file"""
    return logging.getLogger(name)
