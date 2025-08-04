import logging
import sys
from pathlib import Path

def setup_logging(log_file: str = "rag_api.log"):
    """Setup logging configuration"""
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if log.hasHandlers():
        log.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    
    return logging.getLogger(__name__)
