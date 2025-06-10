"""Utility functions for CSV operations."""

import logging
from pathlib import Path
from typing import Union, Optional
import os

def check_file_exists(file_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> bool:
    """Check if a file exists.
    
    Args:
        file_path (Union[str, Path]): Path to the file to check
        logger (Optional[logging.Logger]): Logger for info messages
        
    Returns:
        bool: True if file exists, False otherwise
    """
    path = Path(file_path)
    exists = os.path.exists(path)
    if logger and not exists:
        logger.info(f"File not found at {path}")
    return exists 