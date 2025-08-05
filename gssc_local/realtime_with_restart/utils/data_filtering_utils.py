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

from gssc_local.realtime_with_restart.export.csv.utils import filter_previously_seen_timestamps, filter_duplicate_timestamps_within_batch
from gssc_local.realtime_with_restart.utils.timestamp_utils import reorder_samples_by_timestamp, validate_sample_rate, validate_no_duplicates, validate_monotonic_order, validate_inter_batch_sample_rate
from gssc_local.realtime_with_restart.export.csv.validation import find_duplicates


def sanitize_data(board_data, board_timestamp_channel, logger, last_saved_timestamp=None, expected_sample_rate=None):
    """Sanitize raw data by removing duplicates, reordering samples, and validating integrity before processing"""
    filtered_data = board_data
    
    if last_saved_timestamp is not None:
        # check for old timestamps
        from gssc_local.realtime_with_restart.export.csv.utils import transform_data_to_rows
        rows = transform_data_to_rows(filtered_data, logger)
        filtered_rows, old_timestamps_removed = filter_previously_seen_timestamps(
            rows, board_timestamp_channel, last_saved_timestamp, logger
        )
        if old_timestamps_removed > 0:
            logger.warning(f"Filtered {old_timestamps_removed} old timestamps")
            # Convert back to original format if rows were filtered
            if filtered_rows:
                import numpy as np
                filtered_data = np.array(filtered_rows).T

    filtered_data, duplicates_removed = filter_duplicate_timestamps_within_batch(
        filtered_data, board_timestamp_channel, logger
    )
    if duplicates_removed > 0:
        logger.warning(f"Filtered {duplicates_removed} duplicate timestamps")

    # Reorder samples by timestamp to ensure monotonic order
    filtered_data, was_reordered = reorder_samples_by_timestamp(
        filtered_data, board_timestamp_channel, logger
    )
    if was_reordered:
        logger.warning("Reordered samples to ensure monotonic timestamp order")

    # Validate inter-batch sample rate using filtered data (only if not first batch)
    if last_saved_timestamp is not None:
        validate_inter_batch_sample_rate(
            last_saved_timestamp, 
            filtered_data, 
            board_timestamp_channel, 
            expected_sample_rate, 
            logger
        )

    # Validate sample rate within the batch
    timestamps = filtered_data[board_timestamp_channel, :]
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

    return filtered_data