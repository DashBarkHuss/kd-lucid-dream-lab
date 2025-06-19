"""
ETD Buffer Manager for handling the elctrode and timestamp buffer management and trimming.

This module provides functionality for managing the electrode_and_timestamp_data buffer,
including trimming, validation, and index tracking to maintain memory efficiency while
preserving data continuity for overlapping epoch processing.

Data Shape Conventions:
- Input data (new_data, brainflow stream data): Shape (n_channels, n_samples)
    - Each row represents a channel
    - Each column represents a time point
    - Example: For 8 channels and 1000 samples: shape (8, 1000)
    - This is the raw data format from the board

- Internal buffer (electrode_and_timestamp_data): List of lists
    - Each inner list represents one channel
    - Each inner list contains values for all time points for that channel
    - Example: For 8 channels and 1000 samples:
        [
            [ch1_t1, ch1_t2, ..., ch1_t1000],  # Channel 1
            [ch2_t1, ch2_t2, ..., ch2_t1000],  # Channel 2
            ...
            [ch8_t1, ch8_t2, ..., ch8_t1000]  # Channel 8
        ]

See individual method docstrings for detailed documentation.
"""

import numpy as np
import logging
from typing import List, Tuple
logger = logging.getLogger(__name__)


class ETDBufferManager:
    """Manages the electrode_and_timestamp_data buffer operations.
    
    This class handles all buffer-related operations for the electrode_and_timestamp_data,
    including trimming, validation, and offset tracking.

    Data Shape Conventions:
        - Input data (new_data, brainflow stream data): Shape (n_channels, n_samples)
            - Each row represents a channel
            - Each column represents a time point
            - Example: For 8 channels and 1000 samples: shape (8, 1000)
            - This is the raw data format from the board
        
        - Internal buffer (electrode_and_timestamp_data): List of lists
            - Each inner list represents one channel
            - Each inner list contains values for all time points for that channel
            - Example: For 8 channels and 1000 samples:
                [
                    [ch1_t1, ch1_t2, ..., ch1_t1000],  # Channel 1
                    [ch2_t1, ch2_t2, ..., ch2_t1000],  # Channel 2
                    ...
                    [ch8_t1, ch8_t2, ..., ch8_t1000]  # Channel 8
                ]
    """
    
    def __init__(self, max_buffer_size: int, timestamp_channel_index: int, channel_count: int, electrode_and_timestamp_channels: List[int]):
        """Initialize the buffer manager.
        
        Args:
            max_buffer_size: Maximum number of data points to maintain in buffer
            timestamp_channel_index: Index of the timestamp channel in the buffer
            channel_count: Number of channels in the buffer
            electrode_and_timestamp_channels: List of channel indices to store in the buffer
        """
        self.max_buffer_size = max_buffer_size
        self.offset = 0  # Track absolute position in the data stream
        self.total_streamed_samples = 0  # Track total samples processed
        self.timestamp_channel_index = timestamp_channel_index
        self.electrode_and_timestamp_channels = electrode_and_timestamp_channels
        self.electrode_and_timestamp_data = [[] for _ in range(channel_count)]
        
    def _get_total_data_points(self) -> int:
        """Get total number of data points in buffer.
        
        Returns:
            Number of data points in buffer (length of any channel's data list)
        """
        return len(self.electrode_and_timestamp_data[0]) if self.electrode_and_timestamp_data else 0
        
    def _get_timestamps(self) -> List[float]:
        """Get timestamps from electrode_and_timestamp_data buffer.
        
        Returns:
            List of timestamps from the buffer (from the timestamp channel)
        """
        return self.electrode_and_timestamp_data[self.timestamp_channel_index]  # Use correct timestamp channel index
        
    def _verify_timestamp_continuity(self, timestamps: List[float]) -> None:
        """Verify timestamp continuity in the buffer.
        
        Args:
            timestamps: List of timestamps to verify
            
        Raises:
            ValueError: If timestamp continuity is broken
        """
        if not timestamps:
            return
            
        # Check for timestamp continuity
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                raise ValueError(
                    f"Timestamp continuity broken at index {i}: "
                    f"{timestamps[i-1]} -> {timestamps[i]}"
                )
                
    def _validate_buffer_after_trim(self, expected_size: int) -> None:
        """Validate the buffer state after trimming.
        
        This method performs comprehensive validation of the buffer state:
        1. Verifies buffer size matches expected size
        2. Checks timestamp continuity
        3. Validates channel synchronization
        4. Ensures offset tracking is correct
        
        Args:
            expected_size: Expected number of data points in buffer
            
        Raises:
            ValueError: If any validation check fails
        """
        # Verify buffer size
        actual_size = self._get_total_data_points()
        if actual_size != expected_size:
            raise ValueError(
                f"Buffer size validation failed: expected {expected_size}, got {actual_size}"
            )
            
        # Verify all channels have the same length
        channel_lengths = [len(channel) for channel in self.electrode_and_timestamp_data]
        if not all(length == expected_size for length in channel_lengths):
            raise ValueError(
                f"Channel synchronization failed: lengths {channel_lengths} should all be {expected_size}"
            )
            
        # Verify timestamp continuity
        timestamps = self._get_timestamps()
        self._verify_timestamp_continuity(timestamps)
        
        # Verify offset tracking
        if self.offset < 0:
            raise ValueError(f"Invalid offset: {self.offset} should be non-negative")
            
        # Verify total streamed samples tracking
        if self.total_streamed_samples < self.offset:
            raise ValueError(
                f"Invalid total samples: {self.total_streamed_samples} "
                f"should be >= offset {self.offset}"
            )
            
    def _update_offset_tracking(self, points_removed: int) -> None:
        """Update the offset tracking to maintain absolute position references.
        
        Args:
            points_removed: Number of points removed from buffer
        """
        self.offset += points_removed
        
    def trim_buffer(self, max_next_expected: int = None) -> None: 
        """Trim the buffer to maintain max_buffer_size, but only if data has been processed.
        
        This method will:
        1. Calculate how many points can be safely removed based on the next expected epoch
        2. Remove only the oldest data points that have been fully processed
        3. Update offset tracking
        4. Validate buffer state after trim
        
        Args:
            max_next_expected: The maximum next epoch start index across all buffers.
                If None or 0, no trimming occurs.
                
        Raises:
            ValueError: If validation fails after trim
        """
        if not self.electrode_and_timestamp_data or max_next_expected is None or max_next_expected <= 0:
            return
            
        current_size = self._get_total_data_points()
        if current_size <= self.max_buffer_size:
            return
        
            
        # Calculate how many points we can safely remove
        # We can only remove points up to the maximum next_expected value

        points_greater_than_max_buffer_size = current_size - self.max_buffer_size

        relative_max_next_expected = self._adjust_index_with_offset(max_next_expected)

        safe_remove_points = min(
            points_greater_than_max_buffer_size,  # How many we want to remove
            relative_max_next_expected       # How many we can safely remove
        )
        
        if safe_remove_points <= 0:
            return
            
        # Remove oldest data points from each channel
        for channel in self.electrode_and_timestamp_data:
            channel[:safe_remove_points] = []
            
        # Update offset tracking - this must happen after removing the data
        self._update_offset_tracking(safe_remove_points)
        
        # Validate buffer state after trim
        self._validate_buffer_after_trim(current_size - safe_remove_points)




    def update_total_streamed_samples(self, new_data) -> None:
        """Update the total number of samples streamed.
        
        Args:
            new_data: Array of shape (n_channels, n_samples) containing the data
                - Each row represents a channel
                - Each column represents a time point
                - The number of samples is extracted from the data shape
        """
        # Extract number of samples from the data shape (second dimension)
        new_samples = len(new_data[0]) if len(new_data) > 0 else 0
        self.total_streamed_samples += new_samples

    def _adjust_index_with_offset(self, index: int, to_etd: bool = True) -> int:
        """Helper method to adjust an index based on the etd_offset.
        
        Args:
            index: The index to adjust. For absolute indices, valid range is [0, total_streamed_samples].
                   The value total_streamed_samples is allowed for exclusive end slicing (e.g., data[start:end]).
            to_etd: If True, convert from absolute to relative index. If False, convert from relative to absolute.
            
        Returns:
            int: The adjusted index
            
        Raises:
            ValueError: If the index is negative or exceeds the total streamed samples
        """
        if index < 0:
            raise ValueError(f"Index cannot be negative, got {index}")
            
        if to_etd and index > self.total_streamed_samples:
            raise ValueError(f"Absolute index {index} exceeds total streamed samples {self.total_streamed_samples}")
            
        if not to_etd and index >= self._get_total_data_points():
            raise ValueError(f"Relative index {index} exceeds buffer size {self._get_total_data_points()}")
            
        if to_etd:
            # When converting from absolute to relative, we need to check if the index is before our offset
            if index < self.offset:
                raise ValueError(f"Absolute index {index} is before buffer start (offset: {self.offset})")
            return index - self.offset
        else:
            return index + self.offset

    def add_data(self, new_data):
        """Add new data to the buffer (expects shape: (n_channels, n_samples)).
        
        Args:
            new_data: Array of shape (n_channels, n_samples) containing the data to add
                - Each row represents a channel
                - Each column represents a time point
                - Example: For 8 channels and 1000 samples: shape (8, 1000)
                - This is the raw data format from the board
        """
        # Validate data shape: must be (n_channels, n_samples) BrainFlow's native format
        if new_data.shape[0] < len(self.electrode_and_timestamp_channels):
            raise ValueError(
                f"Input data has {new_data.shape[0]} channels but we need {len(self.electrode_and_timestamp_channels)} "
                f"channels to select from. Expected data in (n_channels, n_samples) format with at least "
                f"{len(self.electrode_and_timestamp_channels)} channels, got shape {new_data.shape}"
            )
        
        # Only store channels we want in the correct order
        for i, channel in enumerate(self.electrode_and_timestamp_channels):
            self.electrode_and_timestamp_data[i].extend(new_data[channel].tolist() if hasattr(new_data[channel], 'tolist') else new_data[channel])
                
        # Update total streamed samples using the dedicated method
        self.update_total_streamed_samples(new_data) 

