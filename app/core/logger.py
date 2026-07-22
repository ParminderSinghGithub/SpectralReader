import logging
import sys
from app.core.config import settings

def get_logger(name: str = "SpectralReader") -> logging.Logger:
    """Construct and configure a structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_str = getattr(logging, settings.LOG_LEVEL, logging.INFO)
        logger.setLevel(level_str)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
