import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Configure logging settings for the application"""
    
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
    # Console handler
    logger.add(
        sys.stderr,
        format=default_config["format"],
        level=default_config["log_level"],
        colorize=True
    )
    
    # File handler for all logs
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