"""Gap detection and reporting functionality for EEG data streams.

This module handles detection of gaps in BrainFlow EEG data streams. Timestamps should be
Unix timestamps (seconds since epoch) as provided by BrainFlow's timestamp column.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

class GapError(Exception):
    """Base exception for gap detection errors."""
    pass

class InvalidTimestampError(GapError):
    """Raised when timestamp data is invalid or malformed."""
    pass

class EmptyTimestampError(GapError):
    """Raised when timestamp array is empty."""
    pass

class InvalidGapThresholdError(GapError):
    """Raised when gap threshold is invalid (e.g., negative or zero)."""
    pass

class InvalidSamplingRateError(GapError):
    """Raised when sampling rate is invalid (e.g., negative or zero)."""
    pass

class InvalidEpochIndicesError(GapError):
    """Raised when epoch start/end indices are invalid."""
    pass

class GapHandler:
    """Detects and reports gaps in BrainFlow EEG data streams.
    
    Identifies discontinuities in timestamp sequences and returns information about
    detected gaps, including their size and location. Timestamps should be Unix timestamps
    (seconds since epoch) as provided by BrainFlow's timestamp column.
    
    Attributes: TODO: Determine if these are needed or should be passed in directy to each method
        expected_interval (float): Expected time interval between samples in seconds
        gap_threshold (float): Threshold for gap detection in seconds
        timestamp_tolerance (float): Tolerance for timestamp deviations
    """
    
    def __init__(self, sampling_rate: float, gap_threshold: float = 2.0):
        """Initialize the GapHandler.
        
        Args:
            sampling_rate: Sampling rate of the EEG data in Hz
            gap_threshold: Threshold for gap detection in seconds (default: 2.0)
            
        Raises:
            InvalidSamplingRateError: If sampling rate is invalid
            InvalidGapThresholdError: If gap threshold is invalid
        """
        if sampling_rate <= 0:
            raise InvalidSamplingRateError(f"Sampling rate must be positive, got {sampling_rate}")
        if gap_threshold <= 0:
            raise InvalidGapThresholdError(f"Gap threshold must be positive, got {gap_threshold}")
            
        self.expected_interval = 1.0 / sampling_rate
        self.gap_threshold = gap_threshold
        self.timestamp_tolerance = self.expected_interval * 0.01  # 1% tolerance
        
    def cleanup(self) -> None:
        """Clean up resources and reset state.
        
        This method should be called when the GapHandler is no longer needed
        to ensure proper resource release and state reset.
        """
        try:
            logger.info("Cleaning up GapHandler resources")
            
            # Reset any internal state if needed
            self.expected_interval = None
            self.gap_threshold = None
            self.timestamp_tolerance = None
            
            logger.info("GapHandler cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during GapHandler cleanup: {e}")
            raise GapError(f"Failed to cleanup GapHandler: {e}")
        
    def _validate_timestamps(self, timestamps: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Validate timestamp array for basic integrity.
        
        Args:
            timestamps: Array of Unix timestamps (seconds since epoch) from BrainFlow
            
        Returns:
            np.ndarray: Validated timestamps as numpy array
            
        Raises:
            EmptyTimestampError: If timestamps array is empty
            InvalidTimestampError: If timestamps are invalid or malformed
        """
        if len(timestamps) == 0:
            raise EmptyTimestampError("Timestamp array cannot be empty")
            
        # Convert to numpy array if it's a list
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)
        elif not isinstance(timestamps, np.ndarray):
            raise InvalidTimestampError(f"Timestamps must be numpy array or list, got {type(timestamps)}")
            
        if not np.issubdtype(timestamps.dtype, np.number):
            raise InvalidTimestampError(f"Timestamps must be numeric, got {timestamps.dtype}")
            
        if np.isnan(timestamps).any():
            raise InvalidTimestampError("Timestamps cannot contain NaN values")
            
        if not np.all(np.diff(timestamps) >= 0):
            raise InvalidTimestampError("Timestamps must be monotonically increasing")
            
        return timestamps
            
    def _validate_epoch_indices(self, timestamps: np.ndarray, start_idx_rel: int, end_idx_rel: int) -> None:
        """Validate epoch indices.
        
        Args:
            timestamps: Array of timestamps
            start_idx: Start index of the epoch
            end_idx: End index of the epoch
            
        Raises:
            InvalidEpochIndicesError: If epoch indices are invalid
        """
        if start_idx_rel < 0 or end_idx_rel > len(timestamps):
            raise InvalidEpochIndicesError(
                f"Epoch indices out of bounds: start_idx={start_idx_rel}, end_idx={end_idx_rel}, "
                f"array_length={len(timestamps)}"
            )
            
        if start_idx_rel >= end_idx_rel:
            raise InvalidEpochIndicesError(
                f"Start index must be less than end index: start_idx={start_idx_rel}, end_idx={end_idx_rel}"
            )
        
    def detect_gap(self, timestamps: np.ndarray, prev_timestamp: Optional[float] = None) -> Tuple[bool, float, Optional[int], Optional[int]]:
        """Detect if there is a gap in the timestamps.
        
        Checks both between chunks and within the current chunk.
        
        Args:
            timestamps: Array of Unix timestamps (seconds since epoch) from BrainFlow
            prev_timestamp: Previous timestamp to compare against (Unix timestamp)
            
        Returns:
            tuple: (has_gap, gap_size, gap_start_idx, gap_end_idx)
            - has_gap: True if a gap was detected
            - gap_size: Size of the largest gap found in seconds
            - gap_start_idx: Start index of the gap (or None if no gap)
            - gap_end_idx: End index of the gap (or None if no gap)
            
        Raises:
            EmptyTimestampError: If timestamps array is empty
            InvalidTimestampError: If timestamps are invalid or malformed
        """
        timestamps = self._validate_timestamps(timestamps)
        
        has_gap = False
        max_gap = 0
        gap_start_idx = None
        gap_end_idx = None

        # Check gap between chunks if we have a previous timestamp
        if prev_timestamp is not None:
            between_chunks_gap = timestamps[0] - prev_timestamp - self.expected_interval
            # Reason: Detect gaps that are strictly greater than the threshold
            if abs(between_chunks_gap) > self.gap_threshold:
                has_gap = True
                max_gap = between_chunks_gap
                gap_start_idx = -1  # -1 indicates gap is between chunks
                gap_end_idx = 0

        # Check gaps within the chunk
        for i in range(1, len(timestamps)):
            interval_deviation = timestamps[i] - timestamps[i-1] - self.expected_interval
            # Reason: Detect gaps that are strictly greater than the threshold
            if abs(interval_deviation) > self.gap_threshold:
                # Reason: Always report the largest gap (by absolute value)
                if not has_gap or abs(interval_deviation) > abs(max_gap):
                    has_gap = True
                    max_gap = interval_deviation
                    gap_start_idx = i-1
                    gap_end_idx = i

        if not has_gap:
            return False, 0, None, None
        return True, max_gap, gap_start_idx, gap_end_idx
        
    def validate_epoch_gaps(self, timestamps: np.ndarray, epoch_start_idx_rel: int, epoch_end_idx_rel: int) -> Tuple[bool, float]:
        """Validate the epoch has no gaps.
        
        Args:
            timestamps: Array of Unix timestamps (seconds since epoch) from BrainFlow
            epoch_start_idx: Start index of the epoch relative in timestamps, not absolute in total streamed data
            epoch_end_idx: End index of the epoch relative in timestamps, not absolute in total streamed data   
            
        Returns:
            tuple: (has_gap, gap_size)
            - has_gap: True if a gap was detected
            - gap_size: Size of the gap in seconds if one was detected, otherwise 0
            
        Raises:
            EmptyTimestampError: If timestamps array is empty
            InvalidTimestampError: If timestamps are invalid or malformed
            InvalidEpochIndicesError: If epoch indices are invalid
        """
        self._validate_timestamps(timestamps)

        # this validation expects the epoch_start_idx and epoch_end_idx to be the indices of the epochs in the timestamps array.
        # if we cut the timestamps array without adjusting the indices then the indices will be incorrect.
        self._validate_epoch_indices(timestamps, epoch_start_idx_rel, epoch_end_idx_rel)
        
        # Get the epoch timestamps
        epoch_timestamps = timestamps[epoch_start_idx_rel:epoch_end_idx_rel]
        
        # Get the previous timestamp if available
        prev_timestamp = timestamps[epoch_start_idx_rel-1] if epoch_start_idx_rel > 0 else None
        
        # Check for gaps
        has_gap, gap_size, _, _ = self.detect_gap(epoch_timestamps, prev_timestamp)
        
        return has_gap, gap_size 