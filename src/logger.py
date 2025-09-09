import logging
from logging.handlers import RotatingFileHandler

import os

def get_logger(name:str, log_file:str):

    """
    Return a logger instance with rotating file + console handler
    Each stage can pass its own log file
    """

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # Create a rotating handler
    file_handler = RotatingFileHandler(
        f"logs/{log_file}",  # e.g., data_ingestion.log
        maxBytes=5_000_000,  # 5MB per file
        backupCount=3        # Keep 3 old files
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Stream handler (console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

     # Avoid adding duplicate handlers if already set up
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

