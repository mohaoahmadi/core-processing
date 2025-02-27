"""Logging configuration module.

This module provides a centralized logging configuration for the application
using the loguru library. It sets up multiple logging handlers with different
levels and formats for console output and file-based logging.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Configure application-wide logging settings.
    
    This function sets up a comprehensive logging system with the following features:
    - Console output with colored formatting
    - File-based logging with rotation
    - Separate error log file
    - Customizable log formats and levels
    
    Args:
        config (Dict[str, Any], optional): Custom logging configuration with keys:
            - log_level: Minimum log level to capture
            - format: Log message format string
            - rotation: When to rotate log files (e.g., "500 MB")
            - retention: How long to keep old logs (e.g., "10 days")
            - compression: Compression format for rotated logs
            
    Note:
        Default Configuration:
        - Log Level: INFO
        - Format: Timestamp | Level | Module:Function:Line - Message
        - Rotation: 500 MB
        - Retention: 10 days
        - Compression: zip
        
        Log files are stored in the 'logs' directory:
        - app.log: All log messages
        - error.log: Error messages only
    """
    
    # Remove default handler
    logger.remove()
    
    # Default configuration
    default_config = {
        "log_level": "INFO",
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        "rotation": "500 MB",
        "retention": "10 days",
        "compression": "zip"
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add handlers
    # Console handler with colored output
    logger.add(
        sys.stderr,
        format=default_config["format"],
        level=default_config["log_level"],
        colorize=True
    )
    
    # File handler for all logs with rotation
    logger.add(
        str(log_dir / "app.log"),
        format=default_config["format"],
        level=default_config["log_level"],
        rotation=default_config["rotation"],
        retention=default_config["retention"],
        compression=default_config["compression"]
    )
    
    # File handler for errors only
    logger.add(
        str(log_dir / "error.log"),
        format=default_config["format"],
        level="ERROR",
        rotation=default_config["rotation"],
        retention=default_config["retention"],
        compression=default_config["compression"],
        filter=lambda record: record["level"].name == "ERROR"
    )
    
    logger.info("Logging system initialized") 