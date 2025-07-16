"""
Logger utility for the Hybrid Collaborative Filtering Recommendation System
Provides standardized logging across all modules
"""

import logging
import os
from datetime import datetime
from config import LOGGING_CONFIG

def setup_logger():
    """Set up the main logger for the application"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOGGING_CONFIG.get('log_file', 'logs/recommender.log'))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
        format=LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(LOGGING_CONFIG.get('log_file', 'logs/recommender.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('recommender')
    logger.info("Logger initialized successfully")
    return logger

def get_logger(name):
    """Get a logger instance for a specific module"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOGGING_CONFIG.get('level', 'INFO')))
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, LOGGING_CONFIG.get('level', 'INFO')))
        
        # Create formatter
        formatter = logging.Formatter(LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        # Also add file handler if log file is specified
        log_file = LOGGING_CONFIG.get('log_file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, LOGGING_CONFIG.get('level', 'INFO')))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger

def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {duration:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

if __name__ == "__main__":
    # Test the logger
    logger = setup_logger()
    logger.info("Logger test successful")
