"""
Batch processing utilities for sleep stage inference from CSV files.

This module provides functions for extracting epochs from CSV files,
running inference in batch mode, and legacy compatibility functions.
"""

import pandas as pd
import numpy as np
from typing import List

from brainflow.board_shim import BoardShim
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import (
    create_numpy_data_with_brainflow_keys
)
from sleep_scoring_toolkit.utils.csv_processing_utils import load_brainflow_csv_raw


def _load_and_parse_csv(csv_path: str, board_id: int):
    """Load CSV data and extract timestamps and EEG channels."""
    csv_data_raw = load_brainflow_csv_raw(csv_path)
    
    # Use BoardShim to get correct timestamp and EEG channel indices
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    
    csv_timestamps = csv_data_raw[timestamp_channel, :]
    eeg_data = csv_data_raw[eeg_channels, :]
    
    return eeg_data, csv_timestamps, eeg_channels


def _find_epoch_boundaries(timestamps: np.ndarray, epoch_start_time: float, sampling_rate: int):
    """Calculate start and end indices for 30-second epoch."""
    start_idx = np.searchsorted(timestamps, epoch_start_time)
    points_per_epoch = 30 * sampling_rate  # 3750 points for 30 seconds at 125 Hz
    end_idx = start_idx + points_per_epoch
    
    # Validate we have enough data
    if end_idx > len(timestamps):
        raise ValueError(f"Not enough data for full epoch. Need {points_per_epoch} points from index {start_idx}")
    
    return start_idx, end_idx


def _extract_raw_epoch_data(eeg_data: np.ndarray, timestamps: np.ndarray, start_idx: int, end_idx: int):
    """Extract raw epoch data and calculate timing information."""
    epoch_data_raw = eeg_data[:, start_idx:end_idx]
    actual_start_time = timestamps[start_idx]
    actual_end_time = timestamps[end_idx-1]
    
    return epoch_data_raw, actual_start_time, actual_end_time


def extract_epoch_from_csv(csv_path: str, epoch_start_time: float, sampling_rate: int, board_id: int):
    """Extract an epoch (30 seconds) of EEG data starting from epoch_start_time.
    
    Args:
        csv_path: Path to the CSV file
        epoch_start_time: Start time for the epoch
        sampling_rate: EEG sampling rate in Hz (must be provided)
        board_id: Board ID for channel configuration
    
    Returns:
        NumPyDataWithBrainFlowDataKey: Structured epoch data with proper channel mapping
        float: Actual start time of extracted epoch
        float: Actual end time of extracted epoch
    """
    eeg_data, timestamps, eeg_channels = _load_and_parse_csv(csv_path, board_id)
    start_idx, end_idx = _find_epoch_boundaries(timestamps, epoch_start_time, sampling_rate)
    epoch_data_raw, actual_start_time, actual_end_time = _extract_raw_epoch_data(eeg_data, timestamps, start_idx, end_idx)
    epoch_data_keyed = create_numpy_data_with_brainflow_keys(epoch_data_raw, eeg_channels)
    return epoch_data_keyed, actual_start_time, actual_end_time


def extract_and_prepare_epoch_data(csv_path, epoch_start_time, epoch_num, sampling_rate, board_id):
    """Extract epoch data from CSV and prepare for inference."""
    try:
        epoch_data_keyed, actual_start_time, actual_end_time = extract_epoch_from_csv(
            csv_path, epoch_start_time, sampling_rate, board_id
        )
        
        # Data is already properly structured and keyed, no additional preparation needed
        return epoch_data_keyed
        
    except Exception as e:
        raise RuntimeError(f"Could not extract epoch {epoch_num}: {e}") from e


