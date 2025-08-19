#!/usr/bin/env python3

"""
Shared logging utilities for the realtime_with_restart module.

This module provides colored logging functionality that can be used
across different main files and components in the system.
"""

import logging


class LogColors:
    """ANSI color codes for terminal output formatting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level."""
    
    def format(self, record):
        """Format the log record with appropriate colors.
        
        Args:
            record: LogRecord object containing the log message
            
        Returns:
            str: Formatted log message with color codes
        """
        # Choose color based on log level
        if record.levelno >= logging.ERROR:
            color = LogColors.RED
        elif record.levelno >= logging.WARNING:
            color = LogColors.YELLOW
        elif record.levelno >= logging.INFO:
            color = LogColors.GREEN
        else:
            color = LogColors.BLUE
            
        # Apply color to the message
        record.msg = f"{color}{record.msg}{LogColors.ENDC}"
        return super().format(record)


def setup_colored_logger(name=__name__, level=logging.INFO, format_string='%(processName)s - %(levelname)s - L%(lineno)s - %(message)s'):
    """Set up a logger with colored output.
    
    Args:
        name (str): Logger name (defaults to current module)
        level: Logging level (defaults to INFO)
        format_string (str): Log message format string
        
    Returns:
        logging.Logger: Configured logger with colored formatter
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(format_string))
    logger.addHandler(console_handler)
    
    return logger