"""
Timestamp utility functions for processing and formatting timestamps.

This module provides utilities for:
- Converting Unix timestamps to human-readable format in HST (Hawaii Standard Time)
- Formatting elapsed time in HH:MM:SS.mmm format
- Calculating elapsed time between timestamps

Extracted from main.py and main_speed_controlled_stream.py for better testability and reusability.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


def format_timestamp(ts: Optional[float]) -> str:
    """Convert Unix timestamp to human-readable format in HST (Hawaii Standard Time).
    
    Args:
        ts (Optional[float]): Unix timestamp to convert. If None, returns "None".
        
    Returns:
        str: Formatted timestamp string in HST with format 'YYYY-MM-DD HH:MM:SS AM/PM HST'
        
    Examples:
        >>> format_timestamp(1640995200.0)  # 2022-01-01 00:00:00 UTC
        '2021-12-31 02:00:00 PM HST'
        >>> format_timestamp(None)
        'None'
    """
    if ts is None:
        return "None"
    
    # Convert Unix timestamp to datetime in UTC
    utc_time = datetime.fromtimestamp(ts, timezone.utc)
    # Convert to Hawaii time (UTC-10)
    hawaii_time = utc_time - timedelta(hours=10)
    return hawaii_time.strftime('%Y-%m-%d %I:%M:%S %p HST')


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in HH:MM:SS.mmm format.
    
    Args:
        seconds (float): Elapsed time in seconds
        
    Returns:
        str: Formatted time string in HH:MM:SS.mmm format
        
    Examples:
        >>> format_elapsed_time(3661.5)
        '01:01:01.500'
        >>> format_elapsed_time(0.0)
        '00:00:00.000'
        >>> format_elapsed_time(45.123)
        '00:00:45.123'
    """
    # Handle negative values by taking absolute value
    abs_seconds = abs(seconds)
    
    # Convert seconds to hours, minutes, and remaining seconds
    hours = int(abs_seconds // 3600)
    minutes = int((abs_seconds % 3600) // 60)
    remaining_seconds = abs_seconds % 60
    
    # Format with 3 decimal places for milliseconds
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:06.3f}"


def calculate_elapsed_time(start_timestamp: float, end_timestamp: float) -> float:
    """Calculate elapsed time between two timestamps.
    
    Args:
        start_timestamp (float): Starting Unix timestamp
        end_timestamp (float): Ending Unix timestamp
        
    Returns:
        float: Elapsed time in seconds (can be negative if end < start)
        
    Examples:
        >>> calculate_elapsed_time(1640995200.0, 1640995261.5)
        61.5
        >>> calculate_elapsed_time(1640995200.0, 1640995200.0)
        0.0
    """
    return end_timestamp - start_timestamp