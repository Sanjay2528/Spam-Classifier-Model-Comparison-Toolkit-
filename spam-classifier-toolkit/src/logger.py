"""
logger.py
---------
Provides a single, reusable logger instance for the whole application.
Centralising logging configuration satisfies the non-functional
"logging/monitoring" requirement and avoids duplicate handlers being
attached every time a module is imported.
"""

import logging
import os
from src import config


def get_logger(name: str = "spam_toolkit") -> logging.Logger:
    """Return a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (avoids duplicate log lines when
        # get_logger() is called from multiple modules).
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(os.path.dirname(config.LOG_FILE_PATH), exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_FILE_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
