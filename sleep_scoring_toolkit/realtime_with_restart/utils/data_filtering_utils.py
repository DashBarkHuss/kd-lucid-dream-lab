"""
Data sanitization utilities for early-stage pipeline processing.

This module provides comprehensive sanitization and validation for raw board data
before it enters the main processing pipeline. Functions include:
- Duplicate timestamp filtering and removal
- Out-of-order sample reordering  
- Sample rate validation
- Inter-batch consistency checking

Extracted from received_stream_data_handler.py for better modularity and reusability.
"""

import numpy as np
from sleep_scoring_toolkit.realtime_with_restart.export.csv.utils import filter_previously_seen_timestamps, filter_duplicate_timestamps_within_batch
from sleep_scoring_toolkit.realtime_with_restart.utils.timestamp_utils import reorder_samples_by_timestamp, validate_sample_rate, validate_no_duplicates, validate_monotonic_order, validate_inter_batch_sample_rate
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import RawBoardDataWithKeys
from sleep_scoring_toolkit.realtime_with_restart.export.csv.validation import find_duplicates


def filter_previously_seen_timestamps_array(data_keyed, timestamp_board_position: int, last_saved_timestamp: float, logger=None):
    """Filter out samples with timestamps less than or equal to the last saved timestamp.
    
    Efficient array-based filtering that avoids expensive transpose operations.
    Array-optimized version of filter_previously_seen_timestamps for RawBoardDataWithKeys objects.
    
    Args:
        data_keyed (RawBoardDataWithKeys): Keyed data object
        timestamp_board_position (int): Board position of the timestamp channel (use BoardShim.get_timestamp_channel())
        last_saved_timestamp (float): Last saved timestamp to compare against
        logger (Optional[logging.Logger]): Logger instance for reporting
        
    Returns:
        Tuple containing:
            - Filtered data (RawBoardDataWithKeys)
            - Number of old samples that were removed
    """
    
    if not isinstance(data_keyed, RawBoardDataWithKeys):
        raise TypeError("Expected RawBoardDataWithKeys object, got {type(data_keyed)}")
    
    # Extract timestamp channel directly (no transpose needed)
    timestamps = data_keyed.get_by_key(timestamp_board_position)
    
    # Find valid sample indices (timestamps > last_saved_timestamp)
    valid_indices = timestamps > last_saved_timestamp
    old_timestamps_removed = len(timestamps) - np.sum(valid_indices)
    
    if old_timestamps_removed == 0:
        return data_keyed, 0
        
    # Filter data using column indices (no transpose needed)
    filtered_data_array = data_keyed.data[:, valid_indices]
    filtered_data_keyed = RawBoardDataWithKeys(filtered_data_array)
        
    return filtered_data_keyed, old_timestamps_removed


def sanitize_data(raw_board_data_keyed, board_timestamp_channel, logger, last_saved_timestamp=None, expected_sample_rate=None):
    """Sanitize raw data by removing duplicates, reordering samples, and validating integrity before processing"""
    sanitized_board_data_keyed = raw_board_data_keyed
    
    if last_saved_timestamp is not None:
        # Filter old timestamps efficiently (no transpose needed)
        sanitized_board_data_keyed, old_timestamps_removed = filter_previously_seen_timestamps_array(
            sanitized_board_data_keyed, board_timestamp_channel, last_saved_timestamp, logger
        )
        if old_timestamps_removed > 0:
            logger.warning(f"Filtered {old_timestamps_removed} old timestamps")

    sanitized_board_data_keyed, duplicates_removed = filter_duplicate_timestamps_within_batch(
        sanitized_board_data_keyed, board_timestamp_channel, logger
    )
    if duplicates_removed > 0:
        logger.warning(f"Filtered {duplicates_removed} duplicate timestamps")

    # Reorder samples by timestamp to ensure monotonic order
    sanitized_board_data_keyed, was_reordered = reorder_samples_by_timestamp(
        sanitized_board_data_keyed, board_timestamp_channel, logger
    )
    if was_reordered:
        logger.warning("Reordered samples to ensure monotonic timestamp order")

    # Validate inter-batch sample rate using filtered data (only if not first batch)
    if last_saved_timestamp is not None:
        validate_inter_batch_sample_rate(
            last_saved_timestamp, 
            sanitized_board_data_keyed, 
            board_timestamp_channel, 
            expected_sample_rate, 
            logger
        )

    # Validate sample rate within the batch
    timestamps = sanitized_board_data_keyed.get_by_key(board_timestamp_channel)
    is_valid, actual_rate = validate_sample_rate(timestamps, expected_sample_rate, logger=logger)
    if not is_valid:
        logger.warning(f"Sample rate validation failed within batch: expected={expected_sample_rate:.2f}Hz, actual={actual_rate:.2f}Hz")

    # Final validation: check no duplicates and proper ordering
    has_no_duplicates, duplicate_count = validate_no_duplicates(timestamps, logger)
    is_ordered = validate_monotonic_order(timestamps, logger)

    # check for duplicates in the entire stream
    duplicates = find_duplicates(timestamps, comparison='exact')
    if duplicates:
        logger.error(f"Duplicates found in the entire stream: {duplicates}")
    
    if not has_no_duplicates or not is_ordered:
        logger.error(f"Final validation failed: duplicates={duplicate_count}, ordered={is_ordered}")

    return sanitized_board_data_keyed