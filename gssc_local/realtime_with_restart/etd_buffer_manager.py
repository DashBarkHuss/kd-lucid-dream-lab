"""
Buffer Manager for handling data buffer management and trimming.

This module provides functionality for managing the electrode_and_timestamp_data buffer,
including trimming, validation, and index tracking to maintain memory efficiency while
preserving data continuity for overlapping epoch processing.
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class ETDBufferManager:
    """Manages the electrode_and_timestamp_data buffer operations.
    
    This class handles all buffer-related operations for the electrode_and_timestamp_data,
    including trimming, validation, and offset tracking.
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
            Number of data points in buffer
        """
        return len(self.electrode_and_timestamp_data[0]) if self.electrode_and_timestamp_data else 0
        
    def _get_timestamps(self) -> List[float]:
        """Get timestamps from electrode_and_timestamp_data buffer.
        
        Returns:
            List of timestamps from the buffer
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
        
    def trim_buffer(self, points_to_remove: int) -> None:
        """Trim the buffer by removing the oldest data points.
        
        Args:
            points_to_remove: Number of points to remove from start of buffer
            
        Raises:
            ValueError: If validation fails after trim
        """
        if not self.electrode_and_timestamp_data or points_to_remove <= 0:
            return
            
        # Remove oldest data points from each channel
        for channel in self.electrode_and_timestamp_data:
            channel[:points_to_remove] = []
            
        # Update offset tracking
        self._update_offset_tracking(points_to_remove)
        
        # Validate buffer state after trim
        expected_size = self._get_total_data_points()
        self._validate_buffer_after_trim(expected_size)

    def update_total_streamed_samples(self, new_samples: int) -> None:
        """Update the total number of samples streamed.
        
        Args:
            new_samples: Number of new samples to add to the total
        """
        self.total_streamed_samples += new_samples 

  
    def _adjust_index_with_offset(self, index: int, to_etd: bool = True) -> int:
        """Helper method to adjust an index based on the etd_offset.
        
        Args:
            index: The index to adjust
            to_etd: If True, convert from absolute to relative index. If False, convert from relative to absolute.
            
        Returns:
            int: The adjusted index
        """
        if to_etd:
            return index - self.offset
        else:
            return index + self.offset

    def _get_max_buffer_size(self, sampling_rate: int) -> int:
        """Get the maximum allowed buffer size in data points.
        
        The buffer must maintain 35 seconds of data to ensure all overlapping epochs
        have access to their required data:
        - Latest unprocessed data: 25-55s (30 seconds)
        - Plus the 5-second step size
        - Total: 35 seconds
        
        Args:
            sampling_rate: The sampling rate in Hz
            
        Returns:
            int: Maximum number of data points allowed in the buffer
        """
        return 35 * sampling_rate 

    def add_data(self, new_data):
        """Add new data to the buffer (expects shape: (n_channels, n_samples)).
        
        Args:
            new_data: Array of shape (n_channels, n_samples) containing the data to add
        """
        # Only store channels we want in the correct order
        for i, channel in enumerate(self.electrode_and_timestamp_channels):
            self.electrode_and_timestamp_data[i].extend(new_data[channel].tolist() if hasattr(new_data[channel], 'tolist') else new_data[channel])
                
        self.total_streamed_samples += len(new_data[0]) if len(new_data) > 0 else 0 