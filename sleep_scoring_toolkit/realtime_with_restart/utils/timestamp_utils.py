"""
Timestamp utility functions for processing and formatting timestamps.

This module provides utilities for:
- Converting Unix timestamps to human-readable format in HST (Hawaii Standard Time)
- Formatting elapsed time in HH:MM:SS.mmm format
- Calculating elapsed time between timestamps

Extracted from main.py and main_speed_controlled_stream.py for better testability and reusability.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import numpy as np
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import RawBoardDataWithKeys


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


def reorder_samples_by_timestamp(data_keyed, timestamp_board_position: int, logger=None):
    """Reorder samples by timestamp to ensure monotonically increasing order.
    
    This function sorts the data samples by their timestamps to fix any out-of-order
    samples that may occur during data streaming or processing.
    
    Args:
        data_keyed (RawBoardDataWithKeys): 2D keyed data where each column is a sample and each row is a channel
        timestamp_board_position (int): Board position of the timestamp channel (use BoardShim.get_timestamp_channel())
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        Tuple containing:
            - Reordered data (RawBoardDataWithKeys)
            - Boolean indicating whether reordering was necessary
    """
    
    if not isinstance(data_keyed, RawBoardDataWithKeys):
        raise TypeError(f"Expected RawBoardDataWithKeys object, got {type(data_keyed)}")
    
    if data_keyed.size == 0:
        return data_keyed, False
    
    timestamps = data_keyed.get_by_key(timestamp_board_position)
    
    # Check if timestamps are already sorted
    is_sorted = np.all(timestamps[:-1] <= timestamps[1:])
    
    if is_sorted:
        if logger:
            logger.debug("Timestamps are already in monotonically increasing order")
        return data_keyed, False
    
    # Get sorting indices
    sorted_indices = np.argsort(timestamps)
    
    # Reorder all channels by timestamp
    reordered_data = RawBoardDataWithKeys(data_keyed.data[:, sorted_indices])
    
    if logger:
        logger.info(f"Reordered {data_keyed.shape[1]} samples by timestamp to ensure monotonic order")
        
    return reordered_data, True


def validate_inter_batch_sample_rate(prev_timestamp, current_timestamps, board_timestamp_channel, expected_sample_rate, logger=None):
    """Validate sample rate between batches using the existing validate_sample_rate function."""
    if prev_timestamp is None or current_timestamps.size == 0:
        return True, None
    
    import numpy as np
    # Create a mini array with just the transition timestamps
    transition_timestamps = np.array([prev_timestamp, current_timestamps[board_timestamp_channel, 0]])
    
    is_valid, actual_rate = validate_sample_rate(transition_timestamps, expected_sample_rate, tolerance=0.5, logger=logger)
    
    if logger and not is_valid:
        logger.warning(f"Inter-batch sample rate validation failed")
    
    return is_valid, actual_rate


def validate_sample_rate(timestamps: np.ndarray, expected_sample_rate: float, tolerance: float = 0.05, logger=None) -> Tuple[bool, float]:
    """Validate that the actual sample rate matches the expected sample rate.
    
    This function calculates the actual sample rate from timestamps and compares it
    to the expected sample rate to detect timing issues in data streams.
    
    Args:
        timestamps (np.ndarray): Array of timestamps in seconds
        expected_sample_rate (float): Expected sample rate in Hz
        tolerance (float): Tolerance as a fraction (e.g., 0.05 = 5% tolerance)
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        Tuple[bool, float]: Tuple containing:
            - Boolean indicating whether sample rate is within tolerance
            - Actual calculated sample rate in Hz
    """
    if len(timestamps) < 2:
        if logger:
            logger.warning("Cannot validate sample rate with fewer than 2 timestamps")
        return True, expected_sample_rate
    
    # Calculate actual sample rate from timestamps
    total_duration = timestamps[-1] - timestamps[0]
    num_intervals = len(timestamps) - 1
    actual_sample_rate = num_intervals / total_duration
    
    # Calculate tolerance bounds
    lower_bound = expected_sample_rate * (1 - tolerance)
    upper_bound = expected_sample_rate * (1 + tolerance)
    
    is_valid = lower_bound <= actual_sample_rate <= upper_bound
    
    if logger:
        if is_valid:
            logger.debug(f"Sample rate validation passed: actual={actual_sample_rate:.2f}Hz, expected={expected_sample_rate:.2f}Hz")
        else:
            logger.warning(f"Sample rate validation failed for {len(timestamps)} timestamps: actual={actual_sample_rate:.2f}Hz, expected={expected_sample_rate:.2f}Hz (tolerance={tolerance*100:.1f}%)")
    
    return is_valid, actual_sample_rate


def validate_no_duplicates(timestamps: np.ndarray, logger=None) -> Tuple[bool, int]:
    """Validate that there are no duplicate timestamps.
    
    Args:
        timestamps (np.ndarray): Array of timestamps
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        Tuple[bool, int]: (has_no_duplicates, duplicate_count)
    """
    unique_count = len(np.unique(timestamps))
    total_count = len(timestamps)
    duplicate_count = total_count - unique_count
    has_no_duplicates = duplicate_count == 0
    
    if logger and not has_no_duplicates:
        logger.error(f"Found {duplicate_count} duplicate timestamps")
    
    return has_no_duplicates, duplicate_count


def validate_monotonic_order(timestamps: np.ndarray, logger=None) -> bool:
    """Validate that timestamps are in monotonically increasing order.
    
    Args:
        timestamps (np.ndarray): Array of timestamps
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        bool: True if timestamps are monotonically increasing
    """
    if len(timestamps) <= 1:
        return True
        
    is_ordered = np.all(timestamps[:-1] <= timestamps[1:])
    
    if logger and not is_ordered:
        logger.error("Timestamps are not in monotonically increasing order")
    
    return is_ordered


def is_sample_rate_consistent_by_intervals(timestamps: np.ndarray, expected_sample_rate: float, tolerance: float = 0.1, logger=None) -> Tuple[bool, float, int]:
    """Check if timestamp intervals match expected sample rate using interval-based method.
    
    This is an alternative validation method from detect_stream_duplicates.py that uses
    the average of individual timestamp intervals rather than total duration.
    
    Args:
        timestamps (np.ndarray): Array of timestamps
        expected_sample_rate (float): Expected sampling rate in Hz
        tolerance (float): Tolerance as fraction (0.1 = 10% tolerance)
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        Tuple[bool, float, int]: Tuple containing:
            - is_consistent: Boolean indicating if sample rate is within tolerance
            - avg_interval: Average interval between consecutive samples
            - zero_intervals: Count of zero intervals (duplicate timestamps)
    """
    if len(timestamps) <= 1:
        return True, 0.0, 0
    
    expected_interval = 1.0 / expected_sample_rate
    intervals = np.diff(timestamps)
    avg_interval = np.mean(intervals)
    zero_intervals = np.sum(np.abs(intervals) < 0.000001)
    
    # Check if average interval is within tolerance
    interval_error = abs(avg_interval - expected_interval) / expected_interval
    is_consistent = interval_error <= tolerance and zero_intervals == 0
    
    if logger:
        if is_consistent:
            logger.debug(f"Sample rate validation (interval-based) passed: avg_interval={avg_interval:.6f}s, expected={expected_interval:.6f}s")
        else:
            logger.warning(f"Sample rate validation (interval-based) failed: avg_interval={avg_interval:.6f}s, expected={expected_interval:.6f}s (tolerance={tolerance*100:.1f}%), zero_intervals={zero_intervals}")
    
    return is_consistent, avg_interval, zero_intervals