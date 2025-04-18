import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler to prevent extremely large log files
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger

def get_performance_logger(name="performance", log_dir=None):
    """
    Setup specialized logger for performance metrics
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        logger: Configured performance logger
    """
    # Default log directory if not specified
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    perf_logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if perf_logger.handlers:
        return perf_logger
    
    perf_logger.setLevel(logging.INFO)
    
    # Create performance-specific formatter (simpler format for metrics)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    perf_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logging
    perf_logger.propagate = False
    
    return perf_logger

def get_error_logger(name="error", log_dir=None):
    """
    Setup specialized logger for error tracking
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        logger: Configured error logger
    """
    # Default log directory if not specified
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    error_logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if error_logger.handlers:
        return error_logger
    
    error_logger.setLevel(logging.ERROR)
    
    # Create detailed formatter for errors
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=10  # Keep more backups for errors
    )
    file_handler.setFormatter(formatter)
    error_logger.addHandler(file_handler)
    
    # Also log to console for immediate visibility
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)
    error_logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    error_logger.propagate = False
    
    return error_logger

def configure_root_logger(log_dir=None, level=logging.INFO):
    """
    Configure the root logger for application-wide logging
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    # Default log directory if not specified
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler with current date in filename
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'app_{today}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info("Logger initialized")

def create_timed_rotating_logger(name, log_dir=None, level=logging.INFO):
    """
    Create a logger that rotates files daily
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        logger: Configured logger with daily rotation
    """
    from logging.handlers import TimedRotatingFileHandler
    
    # Default log directory if not specified
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add timed rotating file handler (rotate at midnight each day)
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        when='midnight',
        interval=1,  # Daily rotation
        backupCount=30  # Keep 30 days of logs
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"  # Append date to rotated files
    logger.addHandler(file_handler)
    
    return logger