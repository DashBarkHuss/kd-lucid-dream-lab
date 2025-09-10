"""Unit tests for the GapHandler class."""

import pytest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sleep_scoring_toolkit.realtime_with_restart.core.gap_handler import GapHandler, GapError, InvalidTimestampError, EmptyTimestampError, InvalidGapThresholdError, InvalidSamplingRateError, InvalidEpochIndicesError

# Test constants
SAMPLING_RATE = 100.0  # Hz
GAP_THRESHOLD = 2.0    # seconds
EXPECTED_INTERVAL = 1.0 / SAMPLING_RATE  # 0.01 seconds
TIMESTAMP_TOLERANCE = EXPECTED_INTERVAL * 0.01  # 1% of expected interval
FLOAT_COMPARISON_TOLERANCE = 0.001  # For comparing floating point numbers
BASE_TIME = 1746193963.801430  # Example BrainFlow timestamp with microsecond precision

@pytest.fixture
def gap_handler():
    """Create a GapHandler instance with standard parameters."""
    return GapHandler(sampling_rate=SAMPLING_RATE, gap_threshold=GAP_THRESHOLD)

def test_initialization():
    """Test GapHandler initialization with valid and invalid parameters."""
    # Test valid initialization
    handler = GapHandler(sampling_rate=SAMPLING_RATE, gap_threshold=GAP_THRESHOLD)
    assert handler.expected_interval == EXPECTED_INTERVAL
    assert handler.gap_threshold == GAP_THRESHOLD
    assert handler.timestamp_tolerance == TIMESTAMP_TOLERANCE

    # Test invalid sampling rate
    with pytest.raises(InvalidSamplingRateError):
        GapHandler(sampling_rate=0.0, gap_threshold=GAP_THRESHOLD)
    with pytest.raises(InvalidSamplingRateError):
        GapHandler(sampling_rate=-SAMPLING_RATE, gap_threshold=GAP_THRESHOLD)

    # Test invalid gap threshold
    with pytest.raises(InvalidGapThresholdError):
        GapHandler(sampling_rate=SAMPLING_RATE, gap_threshold=0.0)
    with pytest.raises(InvalidGapThresholdError):
        GapHandler(sampling_rate=SAMPLING_RATE, gap_threshold=-GAP_THRESHOLD)

def test_validate_timestamps(gap_handler):
    """Test timestamp validation functionality."""
    # Test valid timestamps with microsecond precision
    valid_timestamps = np.array([
        BASE_TIME,
        BASE_TIME + EXPECTED_INTERVAL,
        2*EXPECTED_INTERVAL + BASE_TIME,
        3*EXPECTED_INTERVAL + BASE_TIME
    ])
    gap_handler._validate_timestamps(valid_timestamps)  # Should not raise

    # Test empty timestamps
    with pytest.raises(EmptyTimestampError):
        gap_handler._validate_timestamps(np.array([]))

    # Test valid list (should be converted to numpy array)
    gap_handler._validate_timestamps([BASE_TIME, BASE_TIME + EXPECTED_INTERVAL])  # Should not raise
    
    # Test invalid type (neither list nor numpy array)
    with pytest.raises(InvalidTimestampError):
        gap_handler._validate_timestamps("not_a_valid_type")

    # Test non-numeric timestamps
    with pytest.raises(InvalidTimestampError):
        gap_handler._validate_timestamps(np.array(['a', 'b', 'c']))

    # Test NaN values
    with pytest.raises(InvalidTimestampError):
        gap_handler._validate_timestamps(np.array([BASE_TIME, np.nan, BASE_TIME + 2*EXPECTED_INTERVAL]))

    # Test non-monotonic timestamps
    with pytest.raises(InvalidTimestampError):
        gap_handler._validate_timestamps(np.array([
            BASE_TIME,
            BASE_TIME + 2*EXPECTED_INTERVAL,
            BASE_TIME + EXPECTED_INTERVAL
        ]))

def test_validate_epoch_indices(gap_handler):
    """Test epoch indices validation."""
    timestamps = np.array([
        BASE_TIME,
        BASE_TIME + EXPECTED_INTERVAL,
        BASE_TIME + 2*EXPECTED_INTERVAL,
        BASE_TIME + 3*EXPECTED_INTERVAL,
        BASE_TIME + 4*EXPECTED_INTERVAL
    ])

    # Test valid indices
    gap_handler._validate_epoch_indices(timestamps, 1, 3)  # Should not raise

    # Test out of bounds indices
    with pytest.raises(InvalidEpochIndicesError):
        gap_handler._validate_epoch_indices(timestamps, -1, 3)
    with pytest.raises(InvalidEpochIndicesError):
        gap_handler._validate_epoch_indices(timestamps, 1, 6)

    # Test invalid index order
    with pytest.raises(InvalidEpochIndicesError):
        gap_handler._validate_epoch_indices(timestamps, 3, 1)

def test_detect_gap(gap_handler):
    """Test gap detection between chunks and within chunks."""
    # Test gap between chunks
    prev_timestamp = BASE_TIME - GAP_THRESHOLD - 0.01 - EXPECTED_INTERVAL
    timestamps = np.array([
        BASE_TIME,
        BASE_TIME + EXPECTED_INTERVAL,
        BASE_TIME + 2*EXPECTED_INTERVAL
    ])
    has_gap, gap_size, start_idx, end_idx = gap_handler.detect_largest_gap(timestamps, prev_timestamp)
    assert has_gap
    assert abs(gap_size - (GAP_THRESHOLD + 0.01)) < FLOAT_COMPARISON_TOLERANCE
    assert start_idx == -1  # -1 indicates gap is between chunks
    assert end_idx == 0

    # Test gap within chunk
    timestamps = np.array([
        BASE_TIME,
        BASE_TIME + EXPECTED_INTERVAL,
        BASE_TIME + 2*EXPECTED_INTERVAL + GAP_THRESHOLD + 0.01
    ])
    has_gap, gap_size, start_idx, end_idx = gap_handler.detect_largest_gap(timestamps)
    assert has_gap
    assert abs(gap_size - (GAP_THRESHOLD + 0.01)) < FLOAT_COMPARISON_TOLERANCE
    assert start_idx == 1
    assert end_idx == 2

def test_validate_epoch_gaps(gap_handler):
    """Test validation of gaps between epochs."""
    # Create timestamps for two 30-second epochs with a gap between them
    epoch_duration = 30.0  # 30 seconds per epoch
    samples_per_epoch = int(epoch_duration * SAMPLING_RATE)  # 3000 samples at 100Hz
    
    # First epoch: 30 seconds of data with microsecond precision
    first_epoch = np.array([BASE_TIME + i * EXPECTED_INTERVAL for i in range(samples_per_epoch)])
    
    # Second epoch: 30 seconds of data, with a gap before it
    second_epoch = np.array([
        BASE_TIME + epoch_duration + GAP_THRESHOLD + 0.01 + i * EXPECTED_INTERVAL 
        for i in range(samples_per_epoch)
    ])
    
    # Combine epochs
    timestamps = np.concatenate([first_epoch, second_epoch])
    
    # Test gap between epochs by checking the second epoch
    # This should detect the gap between the end of first epoch and start of second epoch
    has_gap, gap_size = gap_handler.validate_epoch_gaps(timestamps, epoch_start_idx_rel=samples_per_epoch, epoch_end_idx_rel=2*samples_per_epoch)
    assert has_gap
    assert abs(gap_size - (GAP_THRESHOLD + 0.01)) < FLOAT_COMPARISON_TOLERANCE

    # Test no gap within first epoch
    has_gap, gap_size = gap_handler.validate_epoch_gaps(timestamps, epoch_start_idx_rel=0, epoch_end_idx_rel=samples_per_epoch)
    assert not has_gap
    assert gap_size == 0

    # Test invalid epoch indices
    with pytest.raises(InvalidEpochIndicesError):
        gap_handler.validate_epoch_gaps(timestamps, epoch_start_idx_rel=samples_per_epoch, epoch_end_idx_rel=samples_per_epoch-1)

def test_cleanup(gap_handler):
    """Test cleanup functionality."""
    gap_handler.cleanup()
    assert gap_handler.expected_interval is None
    assert gap_handler.gap_threshold is None
    assert gap_handler.timestamp_tolerance is None 

if __name__ == '__main__':
    print("\nRunning tests directly...")
    pytest.main([__file__, '-v'])
