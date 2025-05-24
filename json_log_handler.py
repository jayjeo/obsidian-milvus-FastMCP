"""
JSON Log Handler for Obsidian-Milvus FastMCP
Ensures all output is properly formatted as JSON to prevent parsing errors in Claude Desktop.
"""

import logging
import sys
import json
from typing import Dict, Any

class JSONLogHandler(logging.Handler):
    """
    A custom log handler that formats all log messages as JSON.
    This prevents non-JSON text from being sent to the Claude Desktop application.
    """
    
    def __init__(self, stream=None):
        """Initialize the handler with an optional stream."""
        super().__init__()
        self.stream = stream or sys.stdout
        
    def emit(self, record):
        """Format the record as JSON and write it to the stream."""
        try:
            msg = self.format(record)
            log_entry = {
                "type": "log",
                "level": record.levelname,
                "message": msg,
                "source": record.name
            }
            json_msg = json.dumps(log_entry)
            self.stream.write(json_msg + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)

def setup_json_logging(log_level_str=None):
    """
    Configure logging to ensure all output is properly formatted as JSON.
    This prevents Claude Desktop from receiving non-JSON text.
    
    Args:
        log_level_str: Optional log level string. If not provided, will be read from config.
    
    Note: This only configures logging and does not redirect stdout/stderr
    to maintain compatibility with the MCP server.
    """
    # If log_level_str is not provided, get it from config
    if log_level_str is None:
        try:
            import config
            log_level_str = getattr(config, 'LOG_LEVEL', 'ERROR')
        except ImportError:
            # Default to ERROR if config can't be imported
            log_level_str = 'ERROR'
    
    # Convert string log level to logging constant
    log_level = getattr(logging, log_level_str, logging.ERROR)
    
    # Create the JSON handler
    json_handler = JSONLogHandler()
    json_handler.setLevel(log_level)
    
    # Create a formatter that just returns the message without additional formatting
    formatter = logging.Formatter('%(message)s')
    json_handler.setFormatter(formatter)
    
    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to prevent duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our JSON handler
    root_logger.addHandler(json_handler)
    
    # Set the root logger level to match config
    root_logger.setLevel(log_level)
    
    # Log a message to indicate we're using JSON logging (using ERROR level to ensure it's shown)
    root_logger.error(f"JSON logging configured for Claude Desktop compatibility (Level: {log_level_str})")
    
    # Set the level for important loggers but don't iterate through all loggers
    # as this might cause issues with loggers that haven't been initialized yet
    logging.getLogger('OptimizedMCP').setLevel(log_level)
    logging.getLogger('MilvusManager').setLevel(log_level)
    logging.getLogger('HNSWOptimizer').setLevel(log_level)
