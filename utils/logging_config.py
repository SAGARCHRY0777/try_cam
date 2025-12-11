import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logger():
    # Create a logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Name file with start time (when logger is created)
    start_time = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"camera_log_{start_time}.log")

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a RotatingFileHandler
    handler = RotatingFileHandler(
        log_filename,
        maxBytes=100 * 1024 * 1024,  # 100 MB per file
        backupCount=10,              # Keep 10 log files max
    )

    # Formatter with filename and thread info
    formatter = logging.Formatter(
    "%(asctime)s | %(filename)s:%(lineno)d | %(funcName)s | %(threadName)s | [%(levelname)s] - %(message)s")

    handler.setFormatter(formatter)

    # Clear existing handlers (avoid duplicates if setup_logger() is called twice)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger
