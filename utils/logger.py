"""
utils/logger.py
---------------
Centralised loguru logger configuration.
Import `logger` from here in any module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from pathlib import Path

_configured = False


def setup_logger(log_level: str = "INFO", log_file: str = "logs/platform.log"):
    global _configured
    if _configured:
        return logger

    logger.remove()
    # Pretty console output
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    # Rotating file log
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    _configured = True
    return logger


# Auto-configure on import
setup_logger()
