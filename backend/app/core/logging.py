"""Logging Configuration for RL Tutor.

Provides formatted and structured console logging.
"""

import logging
import sys


def setup_logger(name: str = "rl_tutor", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger instance.

    Args:
        name: Name of the logger.
        level: Logging verbosity level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
