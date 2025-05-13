"""
Unit tests for CSVManager class.

This module contains test cases for:
- CSV writing functionality
- Validation logic
- Error handling
- Data integrity checks
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
import unittest
from brainflow.board_shim import BoardShim

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gssc_local.realtime_with_restart.export.csv_manager import CSVManager, CSVExportError, CSVValidationError, CSVDataError, CSVFormatError

class MockBoardShim:
    """Mock BoardShim for testing."""
    def __init__(self, timestamp_channel=0):
        self._timestamp_channel = timestamp_channel
        
    def get_timestamp_channel(self, board_id):
        return self._timestamp_channel
        
    def get_board_id(self):
        return 0

@pytest.fixture
def csv_manager():
    """Fixture providing a CSVManager instance with timestamp index 0."""
    return CSVManager(board_shim=MockBoardShim(timestamp_channel=0))

@pytest.fixture
def csv_manager_large_index():
    """Fixture providing a CSVManager instance with a large timestamp index."""
    return CSVManager(board_shim=MockBoardShim(timestamp_channel=2))  # Using index 2 to test invalid data

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing in (channels, samples) shape."""
    # Create sample data with timestamps in first row (not column)
    timestamps = np.arange(0, 10, 0.1)  # 0 to 9.9 in steps of 0.1
    data = np.vstack([
        timestamps,
        np.sin(timestamps),  # Some sample signal
        np.cos(timestamps)   # Another sample signal
    ])
    return data

@pytest.fixture
def temp_csv_path():
    """Fixture providing a temporary CSV file path."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        return f.name

def test_init_valid():
    """Test valid initialization of CSVManager."""
    manager = CSVManager(board_shim=MockBoardShim(timestamp_channel=0))
    assert manager.saved_data == []
    assert manager.output_csv_path is None
    assert manager.last_saved_timestamp is None
    assert manager.board_shim is not None

def test_init_invalid():
    """Test invalid initialization of CSVManager."""
    # No invalid init for board_shim, so just check for None
    manager = CSVManager()
    assert manager.board_shim is None

def test_validate_data_shape_valid(csv_manager, sample_data):
    """Test data shape validation with valid data."""
    csv_manager._validate_data_shape(sample_data)
    # No exception should be raised

def test_validate_data_shape_invalid(csv_manager, csv_manager_large_index):
    """Test data shape validation with invalid data."""
    # Test with non-numpy array
    with pytest.raises(CSVDataError):
        csv_manager._validate_data_shape([[1, 2, 3], [4, 5, 6]])
    # Test with 1D array
    with pytest.raises(CSVDataError):
        csv_manager._validate_data_shape(np.array([1, 2, 3]))
    # Test with NaN values
    data_with_nan = np.array([[1, 2], [np.nan, 4]])
    with pytest.raises(CSVDataError):
        csv_manager._validate_data_shape(data_with_nan)

def test_save_new_data_initial(csv_manager, sample_data):
    """Test saving initial data."""
    result = csv_manager.save_new_data(sample_data, is_initial=True)
    assert result is True
    assert len(csv_manager.saved_data) == sample_data.shape[1]
    assert csv_manager.last_saved_timestamp == sample_data[0, -1]

def test_save_new_data_subsequent(csv_manager, sample_data):
    """Test saving subsequent data."""
    # Save initial data
    csv_manager.save_new_data(sample_data, is_initial=True)
    initial_length = len(csv_manager.saved_data)
    # Create new data with some overlap (last 5 samples)
    new_data = sample_data[:, -5:]
    result = csv_manager.save_new_data(new_data)
    assert result is True
    # Should add new rows since they have different timestamps
    assert len(csv_manager.saved_data) > initial_length

def test_save_to_csv(csv_manager, sample_data, temp_csv_path):
    """Test saving data to CSV file."""
    # Save some data first
    csv_manager.save_new_data(sample_data, is_initial=True)
    # Save to CSV
    result = csv_manager.save_to_csv(temp_csv_path)
    assert result is True
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    # Compare only the first 3 columns (original data) with higher tolerance
    assert np.allclose(saved_data[:, :3], sample_data.T, rtol=1e-5, atol=1e-5)

def test_validate_saved_csv_format(csv_manager, sample_data, temp_csv_path):
    """Test CSV format validation."""
    # Save data to CSV
    csv_manager.save_new_data(sample_data, is_initial=True)
    csv_manager.save_to_csv(temp_csv_path)
    # Create a reference CSV with same format
    ref_path = temp_csv_path + '.ref'
    np.savetxt(ref_path, sample_data.T, delimiter='\t', fmt='%.6f')
    # Test validation
    result = csv_manager.validate_saved_csv_format(ref_path)
    assert result is True
    # Clean up
    os.remove(ref_path)

def test_validate_saved_csv_matches_original_source(csv_manager, sample_data, temp_csv_path):
    """Test validation against original source."""
    # Save data to CSV
    csv_manager.save_new_data(sample_data, is_initial=True)
    csv_manager.save_to_csv(temp_csv_path)
    # Create a reference CSV
    ref_path = temp_csv_path + '.ref'
    np.savetxt(ref_path, sample_data.T, delimiter='\t', fmt='%.6f')
    # Test validation
    result = csv_manager.validate_saved_csv_matches_original_source(ref_path)
    assert result is True
    # Clean up
    os.remove(ref_path)

def test_add_sleep_stage_to_csv(csv_manager, sample_data):
    """Test adding sleep stage data."""
    # Save initial data
    csv_manager.save_new_data(sample_data, is_initial=True)
    # Add sleep stage data
    sleep_stage = 2.0
    buffer_id = 1.0
    epoch_idx = 0
    csv_manager.add_sleep_stage_to_csv(sleep_stage, buffer_id, epoch_idx)
    # Verify the data was added correctly
    assert csv_manager.saved_data[epoch_idx][-2] == sleep_stage
    assert csv_manager.saved_data[epoch_idx][-1] == buffer_id

def test_add_sleep_stage_invalid(csv_manager, sample_data):
    """Test adding invalid sleep stage data."""
    # Save initial data
    csv_manager.save_new_data(sample_data, is_initial=True)
    
    # Test invalid epoch index (out of bounds)
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_csv(2.0, 1.0, len(csv_manager.saved_data) + 1)
    
    # Test invalid sleep stage type
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_csv("invalid", 1.0, 0)
    
    # Test invalid buffer ID type
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_csv(2.0, "invalid", 0)

def test_validate_timestamp_continuity(csv_manager):
    """Test timestamp continuity validation."""
    # Test valid timestamps
    valid_timestamps = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert csv_manager._validate_timestamp_continuity(valid_timestamps) is True
    
    # Test non-monotonic timestamps
    non_monotonic = pd.Series([1.0, 3.0, 2.0, 4.0, 5.0])
    assert csv_manager._validate_timestamp_continuity(non_monotonic) is False
    
    # Test timestamps with NaN
    with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    assert csv_manager._validate_timestamp_continuity(with_nan) is False

def test_validate_file_path(csv_manager):
    """Test file path validation."""
    # Test valid path
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_path = os.path.join(temp_dir, "test.csv")
        result = csv_manager._validate_file_path(valid_path)
        assert isinstance(result, Path)
    
    # Test invalid path
    with pytest.raises(CSVExportError):
        csv_manager._validate_file_path("/nonexistent/path/test.csv")

def test_error_handling(csv_manager, temp_csv_path):
    """Test error handling in various scenarios."""
    # Test saving to CSV without data
    with pytest.raises(CSVExportError):
        csv_manager.save_to_csv(temp_csv_path)
    
    # Test validating CSV without saving
    with pytest.raises(CSVValidationError):
        csv_manager.validate_saved_csv_format(temp_csv_path)
    
    # Test validating against original without saving
    with pytest.raises(CSVValidationError):
        csv_manager.validate_saved_csv_matches_original_source(temp_csv_path) 

def test_cleanup():
    """Test that cleanup properly resets all state."""
    # Create a CSVManager with some initial state
    manager = CSVManager(board_shim=MockBoardShim(timestamp_channel=0))
    
    # Set up some initial state
    manager.saved_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    manager.last_saved_timestamp = 123.456
    manager.output_csv_path = "test.csv"
    
    # Verify initial state
    assert len(manager.saved_data) == 2
    assert manager.last_saved_timestamp == 123.456
    assert manager.output_csv_path == "test.csv"
    
    # Call cleanup
    manager.cleanup()
    
    # Verify state is reset
    assert len(manager.saved_data) == 0
    assert manager.last_saved_timestamp is None
    assert manager.output_csv_path is None

if __name__ == '__main__':
    print("\nRunning tests directly...")
    pytest.main([__file__, '-v'])

# python -m pytest gssc_local/realtime_with_restart/export/test_csv_manager.py -v