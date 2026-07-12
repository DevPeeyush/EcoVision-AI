import torch
import logging
from configs.paths import LOGS_DIR

# Device Selection (GPU Support with CPU Fallback)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging Configuration
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_LEVEL = logging.INFO

def configure_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance writing to both console and log files."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(LOG_FILE)
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger
