"""
Unit tests for CSVManager class.

This module contains test cases for:
- CSV writing functionality
- Validation logic
- Error handling
- Data integrity checks

Note: The backward compatibility tests verify that deprecated methods still work,
but their behavior has changed to use the new buffer management system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
import unittest
import warnings
import logging
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.board_shim import BrainFlowInputParams

from gssc_local.tests.test_utils import create_brainflow_test_data
from ..realtime_with_restart.export.csv.test_utils import compare_csv_files
from ..realtime_with_restart.export.csv.validation import validate_file_path
from ..realtime_with_restart.export.csv.exceptions import (
    CSVExportError, CSVValidationError, CSVDataError, CSVFormatError,
    MissingOutputPathError, BufferError, BufferOverflowError,
    BufferStateError, BufferValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration of the root logger
)

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gssc_local.realtime_with_restart.export.csv.manager import (
    CSVManager, CSVExportError, CSVValidationError, CSVDataError, 
    CSVFormatError, BufferValidationError, BufferOverflowError,
    BufferStateError, BufferError, MissingOutputPathError
)

@pytest.fixture
def csv_manager():
    """Fixture providing a CSVManager instance with timestamp index 0."""
    input_params = BrainFlowInputParams()
    board_shim = BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params)
    return CSVManager(board_shim=board_shim)

@pytest.fixture
def csv_manager_large_index():
    """Fixture providing a CSVManager instance with a large timestamp index."""
    input_params = BrainFlowInputParams()
    board_shim = BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params)
    return CSVManager(board_shim=board_shim)  # Using index 2 to test invalid data

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing in (channels, samples) shape."""
    # Create 10 seconds of realistic BrainFlow test data
    data, metadata = create_brainflow_test_data(
        duration_seconds=10.0,  # 10 seconds
        sampling_rate=125,      # 125 Hz (Cyton Daisy standard)
        add_noise=False,        # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD
    )
    return data.T  # Transpose to match expected (channels, samples) shape

@pytest.fixture
def temp_csv_path():
    """Fixture providing a temporary CSV file path."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Clean up the file after the test
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_init_valid():
    """Test valid initialization of CSVManager."""
    input_params = BrainFlowInputParams()
    manager = CSVManager(board_shim=BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params))
    assert manager.main_csv_buffer == []
    assert manager.main_csv_path is None
    assert manager.last_saved_timestamp is None
    assert manager.board_shim is not None

def test_init_invalid():
    """Test invalid initialization of CSVManager."""
    # No invalid init for board_shim, so just check for None
    manager = CSVManager()
    assert manager.board_shim is None

def test_validate_data_shape_valid(csv_manager, sample_data):
    """Test data shape validation with valid data."""
    from gssc_local.realtime_with_restart.export.csv.validation import validate_data_shape
    validate_data_shape(sample_data)
    # No exception should be raised

def test_validate_data_shape_invalid(csv_manager, csv_manager_large_index):
    """Test data shape validation with invalid data."""
    from gssc_local.realtime_with_restart.export.csv.validation import validate_data_shape
    # Test with non-numpy array
    with pytest.raises(CSVDataError):
        validate_data_shape([[1, 2, 3], [4, 5, 6]])
    # Test with 1D array
    with pytest.raises(CSVDataError):
        validate_data_shape(np.array([1, 2, 3]))
    # Test with NaN values
    data_with_nan = np.array([[1, 2], [np.nan, 4]])
    with pytest.raises(CSVDataError):
        validate_data_shape(data_with_nan)

def test_validate_saved_csv_matches_original_source(csv_manager, sample_data, temp_csv_path):
    """Test validation against original source."""
    print(f"\nDEBUG: sample_data shape: {sample_data.shape}")
    print(f"DEBUG: sample_data first row: {sample_data[0, :5]}")  # First 5 elements of first row

    # Save data to CSV
    print(f"DEBUG: Before add_data_to_buffer - main_csv_path: {csv_manager.main_csv_path}")
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    print(f"DEBUG: After add_data_to_buffer - main_csv_path: {csv_manager.main_csv_path}")
    
    # Set both paths
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    print(f"DEBUG: After setting paths - main_csv_path: {csv_manager.main_csv_path}")
    
    result = csv_manager.save_all_data()
    print(f"DEBUG: After save_all_data - result: {result}")
    print(f"DEBUG: After save_all_data - main_csv_path: {csv_manager.main_csv_path}")
    assert result is True
    csv_manager.cleanup(reset_paths=False)  # Don't reset paths
    print(f"DEBUG: After cleanup - main_csv_path: {csv_manager.main_csv_path}")

    # Create a reference CSV
    ref_path = temp_csv_path + '.ref'
    np.savetxt(ref_path, sample_data.T, delimiter='\t', fmt='%.6f')
    print(f"DEBUG: Created reference CSV at: {ref_path}")

    # Print contents of both files
    print("\nDEBUG: Contents of saved CSV:")
    with open(temp_csv_path, 'r') as f:
        saved_lines = f.readlines()
        print(f"First 3 lines of saved CSV:\n{''.join(saved_lines[:3])}")

    print("\nDEBUG: Contents of reference CSV:")
    with open(ref_path, 'r') as f:
        ref_lines = f.readlines()
        print(f"First 3 lines of reference CSV:\n{''.join(ref_lines[:3])}")

    # Test validation
    result = compare_csv_files(temp_csv_path, ref_path)
    assert result.matches, f"CSV comparison failed: {result.error_message}"
    assert result.line_count_matches, f"Line count mismatch: expected {result.expected_line_count}, got {result.actual_line_count}"
    assert not result.mismatched_lines, f"Found {len(result.mismatched_lines)} mismatched lines"

    # Clean up reference file
    os.remove(ref_path)

def test_add_sleep_stage_to_csv_buffer(csv_manager, sample_data):
    """Test adding sleep stage data."""
    # Save initial data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    # Add sleep stage data
    sleep_stage = 2.0
    buffer_id = 1.0
    timestamp_start = 0.0  # Start at beginning of data
    timestamp_end = 30.0   # 30-second epoch
    csv_manager.add_sleep_stage_to_sleep_stage_csv(sleep_stage, buffer_id, timestamp_start, timestamp_end)
    # Verify the data was added correctly to sleep stage buffer
    assert len(csv_manager.sleep_stage_buffer) == 1
    entry = csv_manager.sleep_stage_buffer[0]
    assert entry[2] == sleep_stage  # sleep_stage
    assert entry[3] == buffer_id    # buffer_id
    assert entry[0] == timestamp_start  # timestamp_start
    assert entry[1] == timestamp_end    # timestamp_end

def test_add_sleep_stage_invalid(csv_manager, sample_data):
    """Test adding invalid sleep stage data."""
    # Save initial data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    
    # Test invalid sleep stage type
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_sleep_stage_csv("invalid", 1.0, 0.0, 30.0)
    
    # Test invalid buffer ID type
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, "invalid", 0.0, 30.0)
    
    # Test invalid timestamp types
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, "invalid", 30.0)
    
    with pytest.raises(CSVDataError):
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 0.0, "invalid")

def test_validate_file_path(csv_manager):
    """Test file path validation."""
    # Test valid path
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_path = os.path.join(temp_dir, "test.csv")
        result = validate_file_path(valid_path)
        assert isinstance(result, Path)
    
    # Test invalid path
    with pytest.raises(CSVExportError):
        validate_file_path("/nonexistent/path/test.csv")

def test_error_handling(csv_manager, temp_csv_path):
    """Test error handling in various scenarios."""

    # Test validating CSV without saving
    # Create an empty file to test validation
    with open(temp_csv_path, 'w') as f:
        f.write('')
    result = compare_csv_files(temp_csv_path, temp_csv_path)
    assert not result.matches
    assert result.error_message == "Both saved and reference CSV files are empty"
    
    # Test validating against original without saving
    # Create an empty reference file
    ref_path = temp_csv_path + '.ref'
    with open(ref_path, 'w') as f:
        f.write('')
    result = compare_csv_files(temp_csv_path, ref_path)
    assert not result.matches
    assert result.error_message == "Both saved and reference CSV files are empty"

def test_cleanup():
    """Test that cleanup properly resets all state."""
    # Create a CSVManager with some initial state
    input_params = BrainFlowInputParams()
    manager = CSVManager(board_shim=BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params))
    
    # Set up some initial state
    manager.main_csv_buffer = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    manager.last_saved_timestamp = 123.456
    manager.main_csv_path = "test.csv"
    
    # Verify initial state
    assert len(manager.main_csv_buffer) == 2
    assert manager.last_saved_timestamp == 123.456
    assert manager.main_csv_path == "test.csv"
    
    # Call cleanup
    manager.cleanup()
    
    # Verify state is reset
    assert len(manager.main_csv_buffer) == 0
    assert manager.last_saved_timestamp is None
    assert manager.main_csv_path is None

def test_buffer_management_no_output_path(csv_manager):
    """Test buffer management when no output path is set."""
    # Set a small buffer size for testing
    csv_manager.main_buffer_size = 5
    
    # Create data that would fill the buffer
    timestamps = np.arange(0, 10, 0.1)
    data = np.vstack([
        timestamps,
        np.sin(timestamps),
        np.cos(timestamps)
    ])
    
    # Try to add data that would exceed buffer size
    with pytest.raises(MissingOutputPathError):
        csv_manager.add_data_to_buffer(data, is_initial=True)

def test_buffer_management_invalid_data(csv_manager, temp_csv_path):
    """Test buffer management with invalid data."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Try to add invalid data (non-numeric)
    invalid_data = np.array([['invalid', 'data'], ['more', 'invalid']], dtype=object)
    with pytest.raises(CSVDataError):
        csv_manager.add_data_to_buffer(invalid_data, is_initial=True)

def test_buffer_management_empty_data(csv_manager, temp_csv_path):
    """Test buffer management with empty data."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Try to add empty data
    empty_data = np.array([[]])
    with pytest.raises(CSVDataError):
        csv_manager.add_data_to_buffer(empty_data, is_initial=True)

def test_add_sleep_stage_to_sleep_stage_csv(csv_manager, sample_data, temp_csv_path):
    """Test adding sleep stage data to sleep stage buffer."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    
    # Add sleep stage data
    sleep_stage = 2.0
    buffer_id = 1.0
    timestamp_start = 100.0
    timestamp_end = 130.0
    csv_manager.add_sleep_stage_to_sleep_stage_csv(sleep_stage, buffer_id, timestamp_start, timestamp_end)
    
    # Verify the data was added to sleep stage buffer
    assert len(csv_manager.sleep_stage_buffer) == 1
    entry = csv_manager.sleep_stage_buffer[0]
    assert entry[2] == sleep_stage  # sleep_stage
    assert entry[3] == buffer_id    # buffer_id
    assert entry[0] == timestamp_start  # timestamp_start
    assert entry[1] == timestamp_end    # timestamp_end

def test_add_sleep_stage_buffer_overflow(csv_manager, sample_data, temp_csv_path):
    """Test automatic save when sleep stage buffer is full."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    
    # Set a small buffer size for testing
    csv_manager.sleep_stage_buffer_size = 2
    
    # Add data until buffer is full
    for i in range(3):  # Add 3 entries to trigger overflow
        timestamp_start = i * 30.0
        timestamp_end = (i + 1) * 30.0
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i, timestamp_start, timestamp_end)
    
    # Verify buffer was saved and cleared
    assert len(csv_manager.sleep_stage_buffer) == 1  # Only the last entry remains
    assert os.path.exists(temp_csv_path)
    
    # Verify file contents using pandas
    saved_data = pd.read_csv(temp_csv_path, delimiter='\t', header=None, skiprows=1)  # Skip header
    print(f"\nLoaded data shape: {saved_data.shape}")
    print(f"Loaded data:\n{saved_data}")
    
    assert len(saved_data) == 2  # First two entries were saved
    assert saved_data[2].iloc[0] == 2.0  # Check sleep stage value
    assert saved_data[3].iloc[0] == 0.0  # Check buffer ID

def test_add_sleep_stage_buffer_overflow_no_path(csv_manager, sample_data):
    """Test buffer overflow handling when no output path is set."""
    # Set a small buffer size for testing
    csv_manager.sleep_stage_buffer_size = 2

    # Add data until buffer is full
    for i in range(2):
        timestamp_start = i * 30.0
        timestamp_end = (i + 1) * 30.0
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i, timestamp_start, timestamp_end)
    
    # Try to add one more entry - this should raise MissingOutputPathError
    # because the buffer is full and no output path is set
    with pytest.raises(MissingOutputPathError, match="Sleep stage buffer is full \(size: 2\) and no output path is set"):
        timestamp_start = 60.0
        timestamp_end = 90.0
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 2, timestamp_start, timestamp_end)
    
    # Verify buffer state after error
    assert len(csv_manager.sleep_stage_buffer) == 2, "Buffer should still contain exactly 2 entries"
    
    # Verify first entry
    assert csv_manager.sleep_stage_buffer[0][0] == 0.0, "First entry start timestamp incorrect"
    assert csv_manager.sleep_stage_buffer[0][1] == 30.0, "First entry end timestamp incorrect"
    assert csv_manager.sleep_stage_buffer[0][2] == 2.0, "First entry sleep stage incorrect"
    assert csv_manager.sleep_stage_buffer[0][3] == 0.0, "First entry buffer ID incorrect"
    
    # Verify second entry
    assert csv_manager.sleep_stage_buffer[1][0] == 30.0, "Second entry start timestamp incorrect"
    assert csv_manager.sleep_stage_buffer[1][1] == 60.0, "Second entry end timestamp incorrect"
    assert csv_manager.sleep_stage_buffer[1][2] == 2.0, "Second entry sleep stage incorrect"
    assert csv_manager.sleep_stage_buffer[1][3] == 1.0, "Second entry buffer ID incorrect"

def test_save_all_and_cleanup_remaining_data(csv_manager, sample_data, temp_csv_path):
    """Test that save_all_and_cleanup handles remaining data correctly."""
    # Add some sleep stage data
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    timestamp_start = 100.0
    timestamp_end = 130.0
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, timestamp_start, timestamp_end)
    
    # Call save_all_and_cleanup
    result = csv_manager.save_all_and_cleanup(temp_csv_path)
    assert result is True
    
    # Verify sleep stage data was saved
    assert os.path.exists(temp_csv_path + '.sleep')
    # Read the sleep stage data, skipping the header row
    saved_sleep_data = np.loadtxt(temp_csv_path + '.sleep', delimiter='\t', skiprows=1, ndmin=2)
    # The data should be a 1x4 array (one row with 4 columns)
    assert saved_sleep_data.shape == (1, 4)
    assert saved_sleep_data[0, 2] == 2.0  # Check sleep stage value
    assert saved_sleep_data[0, 3] == 1.0  # Check buffer ID
    assert saved_sleep_data[0, 0] == timestamp_start  # Check start timestamp
    assert saved_sleep_data[0, 1] == timestamp_end    # Check end timestamp

def test_save_all_and_cleanup_proper_cleanup(csv_manager, sample_data, temp_csv_path):
    """Test that save_all_and_cleanup performs proper cleanup."""
    # Add some sleep stage data
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    timestamp_start = 100.0
    timestamp_end = 130.0
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, timestamp_start, timestamp_end)
    
    # Call save_all_and_cleanup
    result = csv_manager.save_all_and_cleanup(temp_csv_path)
    assert result is True
    
    # Verify all buffers are cleared
    assert len(csv_manager.main_csv_buffer) == 0
    assert len(csv_manager.sleep_stage_buffer) == 0
    
    # Verify state is reset
    assert csv_manager.last_saved_timestamp is None
    assert csv_manager.main_csv_path is None
    assert csv_manager.sleep_stage_csv_path is None

def test_save_all_and_cleanup_method(csv_manager, sample_data, temp_csv_path):
    """Test the save_all_and_cleanup convenience method.
    
    This method combines save_all_data() and cleanup() into a single operation.
    """
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Save some data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    
    # Add sleep stage data
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 0.0, 30.0)
    
    # Test that the method works correctly
    result = csv_manager.save_all_and_cleanup()
    assert result is True
    
    # Verify cleanup was performed
    assert len(csv_manager.main_csv_buffer) == 0
    assert len(csv_manager.sleep_stage_buffer) == 0
    assert csv_manager.last_saved_timestamp is None
    assert csv_manager.main_csv_path is None
    assert csv_manager.sleep_stage_csv_path is None

def test_save_all_data(csv_manager, sample_data, temp_csv_path):
    """Test saving all data without cleanup.
    Run pytest with -s to see debug output.
    """
    # Add some data to the main buffer
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)

    # Add some sleep stage data
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    timestamp_start = 100.0
    timestamp_end = 130.0
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, timestamp_start, timestamp_end)

    # Set the main path
    csv_manager.main_csv_path = temp_csv_path

    # Call save_all_data
    result = csv_manager.save_all_data()
    assert result is True

    # Verify files exist and have correct content
    assert os.path.exists(temp_csv_path)
    assert os.path.exists(temp_csv_path + '.sleep')

    # Verify main CSV content
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) > 0

    # Verify sleep stage CSV content using pandas
    sleep_stage_data = pd.read_csv(temp_csv_path + '.sleep', delimiter='\t')
    assert len(sleep_stage_data) == 1
    assert sleep_stage_data['sleep_stage'].iloc[0] == 2.0
    assert sleep_stage_data['buffer_id'].iloc[0] == 1.0
    assert sleep_stage_data['timestamp_start'].iloc[0] == timestamp_start
    assert sleep_stage_data['timestamp_end'].iloc[0] == timestamp_end

def test_cleanup_with_path_reset(csv_manager, sample_data, temp_csv_path):
    """Test cleanup with path reset."""
    # Set up some initial state
    csv_manager.main_csv_buffer = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    csv_manager.last_saved_timestamp = 123.456
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    
    # Call cleanup with default reset_paths=True
    csv_manager.cleanup()
    
    # Verify state is reset including paths
    assert len(csv_manager.main_csv_buffer) == 0
    assert csv_manager.last_saved_timestamp is None
    assert csv_manager.main_csv_path is None
    assert csv_manager.sleep_stage_csv_path is None

def test_cleanup_without_path_reset(csv_manager, sample_data, temp_csv_path):
    """Test cleanup without path reset."""
    # Set up some initial state
    csv_manager.main_csv_buffer = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    csv_manager.last_saved_timestamp = 123.456
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    
    # Call cleanup with reset_paths=False
    csv_manager.cleanup(reset_paths=False)
    
    # Verify state is reset but paths are preserved
    assert len(csv_manager.main_csv_buffer) == 0
    assert csv_manager.last_saved_timestamp is None
    assert csv_manager.main_csv_path == temp_csv_path
    assert csv_manager.sleep_stage_csv_path == temp_csv_path + '.sleep'

def test_save_main_buffer_to_csv(csv_manager, sample_data, temp_csv_path):
    """Test saving data to CSV file."""
    # Save some data first
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    # Set both paths
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    # Save to CSV
    result = csv_manager.save_main_buffer_to_csv()
    assert result is True
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    # Compare all channels with exact equality
    np.testing.assert_array_equal(saved_data, sample_data.T)

def test_save_main_buffer_to_csv_uses_new_methods(csv_manager, sample_data, temp_csv_path):
    """Test that save_main_buffer_to_csv uses the new methods correctly."""
    # Save initial data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)

    # Add some sleep stage data
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)

    # Set the main path
    csv_manager.main_csv_path = temp_csv_path

    # Call save_main_buffer_to_csv for main data
    result = csv_manager.save_main_buffer_to_csv()
    assert result is True

    # Call save_sleep_stages_to_csv for sleep stage data
    result = csv_manager.save_sleep_stages_to_csv()
    assert result is True

    # Verify files exist and have correct content
    assert os.path.exists(temp_csv_path)
    assert os.path.exists(temp_csv_path + '.sleep')

    # Verify main CSV content
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) > 0
    # Compare all channels with exact equality
    np.testing.assert_array_equal(saved_data, sample_data.T)

    # Verify sleep stage CSV content using pandas
    sleep_stage_data = pd.read_csv(temp_csv_path + '.sleep', delimiter='\t', header=None, skiprows=1,
                                 names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
    assert len(sleep_stage_data) == 1
    assert sleep_stage_data['sleep_stage'].iloc[0] == 2.0
    assert sleep_stage_data['buffer_id'].iloc[0] == 1.0
    assert sleep_stage_data['timestamp_start'].iloc[0] == 100.0
    assert sleep_stage_data['timestamp_end'].iloc[0] == 130.0

def test_save_main_buffer_to_csv_first_write(csv_manager, sample_data, temp_csv_path):
    """Test first write operation of save_main_buffer_to_csv."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Add data to buffer
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    
    # Save incrementally
    result = csv_manager.save_main_buffer_to_csv()
    
    assert result is True
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    # Compare all channels with exact equality
    np.testing.assert_array_equal(saved_data, sample_data.T)

def test_save_main_buffer_to_csv_append(csv_manager, sample_data, temp_csv_path):
    """Test append operation of save_main_buffer_to_csv."""
    # Set both CSV paths
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'

    # First add data to main buffer
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()

    # Add sleep stage data
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)
    csv_manager.save_sleep_stages_to_csv()

    # Add more sleep stage data
    csv_manager.add_sleep_stage_to_sleep_stage_csv(3.0, 2.0, 200.0, 230.0)
    csv_manager.save_sleep_stages_to_csv()

    # Verify file exists and has correct content
    assert os.path.exists(csv_manager.sleep_stage_csv_path)
    df = pd.read_csv(csv_manager.sleep_stage_csv_path, delimiter='\t', header=None, skiprows=1,
                    names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
    assert len(df) == 2
    assert df['sleep_stage'].tolist() == [2.0, 3.0]
    assert df['buffer_id'].tolist() == [1.0, 2.0]

def test_save_main_buffer_to_csv_no_output_path(csv_manager, sample_data):
    """Test save_main_buffer_to_csv with no output path set."""
    # Add data to buffer
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    
    # Try to save without output path
    with pytest.raises(CSVExportError):
        csv_manager.save_main_buffer_to_csv()

def test_save_main_buffer_to_csv_invalid_data(csv_manager, temp_csv_path):
    """Test save_main_buffer_to_csv with invalid data."""
    # Set invalid data
    csv_manager.main_csv_path = temp_csv_path
    
    # Get number of channels from BrainFlow
    num_channels = BoardShim.get_num_rows(BoardIds.CYTON_DAISY_BOARD)
    
    # Add invalid data with correct number of columns but invalid values
    invalid_data = np.array([['invalid'] * num_channels, ['data'] * num_channels], dtype=object)
    csv_manager.main_csv_buffer = invalid_data.tolist()
    
    # Try to save invalid data
    with pytest.raises(CSVDataError):
        csv_manager.save_main_buffer_to_csv()

def test_save_main_buffer_to_csv_empty_data(csv_manager, temp_csv_path):
    """Test save_main_buffer_to_csv with empty buffer."""
    print("\n=== Testing save_main_buffer_to_csv with empty buffer ===")
    print(f"Initial temp_csv_path exists: {os.path.exists(temp_csv_path)}")
    
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    print(f"Set main_csv_path to: {temp_csv_path}")
    print(f"Current main_csv_buffer length: {len(csv_manager.main_csv_buffer)}")
    
    # Try to save empty buffer
    result = csv_manager.save_main_buffer_to_csv()
    print(f"save_main_buffer_to_csv result: {result}")
    print(f"Final temp_csv_path exists: {os.path.exists(temp_csv_path)}")
    
    assert result is True  # Should succeed but do nothing
    assert os.path.exists(temp_csv_path)  # File should be created
    assert os.path.getsize(temp_csv_path) == 0  # File should be empty
    print("=== End test ===\n")

def test_save_sleep_stages_to_csv_first_write(csv_manager, sample_data, temp_csv_path):
    """Test first write operation of save_sleep_stages_to_csv."""
    print("\n=== Debug test_save_sleep_stages_to_csv_first_write ===")
    
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    print(f"Set sleep_stage_csv_path to: {temp_csv_path}")
    
    # Add data to main buffer first
    print(f"Adding main data to buffer...")
    print(f"Sample data shape: {sample_data.shape}")
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    print(f"Main buffer length after adding data: {len(csv_manager.main_csv_buffer)}")
    
    # Add sleep stage data to buffer
    print(f"Adding sleep stage data...")
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)
    print(f"Sleep stage buffer length: {len(csv_manager.sleep_stage_buffer)}")
    
    # Save sleep stages
    print(f"Saving sleep stages...")
    result = csv_manager.save_sleep_stages_to_csv()
    print(f"Save result: {result}")
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    # Read CSV without expecting a header
    saved_data = pd.read_csv(temp_csv_path, delimiter='\t', header=None, skiprows=1,
                            names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
    print(f"Saved data shape: {saved_data.shape}")
    print(f"Saved data columns: {saved_data.columns.tolist()}")
    assert len(saved_data) == 1
    assert saved_data['sleep_stage'].iloc[0] == 2.0
    assert saved_data['buffer_id'].iloc[0] == 1.0
    
    # Verify buffer is cleared
    assert len(csv_manager.sleep_stage_buffer) == 0
    print("=== End test ===\n")

def test_save_sleep_stages_to_csv_append(csv_manager, temp_csv_path):
    """Test append operation of save_sleep_stages_to_csv."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path

    # Add sleep stage data
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)
    csv_manager.add_sleep_stage_to_sleep_stage_csv(3.0, 2.0, 200.0, 230.0)

    # Save sleep stages
    assert csv_manager.save_sleep_stages_to_csv()

    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    df = pd.read_csv(temp_csv_path, delimiter='\t', header=None, skiprows=1,
                    names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
    assert len(df) == 2
    assert df['sleep_stage'].tolist() == [2.0, 3.0]
    assert df['buffer_id'].tolist() == [1.0, 2.0]
    assert df['timestamp_start'].tolist() == [100.0, 200.0]
    assert df['timestamp_end'].tolist() == [130.0, 230.0]

def test_save_sleep_stages_to_csv_no_output_path(csv_manager, sample_data):
    """Test save_sleep_stages_to_csv with no output path set."""
    # Add sleep stage data to buffer
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)
    
    # Try to save without output path
    with pytest.raises(CSVExportError):
        csv_manager.save_sleep_stages_to_csv()

def test_save_sleep_stages_to_csv_invalid_data(csv_manager, temp_csv_path):
    """Test save_sleep_stages_to_csv with invalid data."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    
    # Add invalid data to buffer
    csv_manager.sleep_stage_buffer = [('invalid', 'data', 'more', 'invalid')]
    
    # Try to save invalid data
    with pytest.raises(CSVDataError):
        csv_manager.save_sleep_stages_to_csv()

def test_save_sleep_stages_to_csv_empty_buffer(csv_manager, temp_csv_path):
    """Test save_sleep_stages_to_csv with empty buffer."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    
    # Try to save empty buffer
    result = csv_manager.save_sleep_stages_to_csv()
    assert result is True  # Should succeed but do nothing
    assert not os.path.exists(temp_csv_path)  # File should not be created

def test_save_sleep_stages_to_csv_maximum_buffer(csv_manager, temp_csv_path):
    """Test save_sleep_stages_to_csv with maximum buffer size."""
    # Set sleep stage CSV path
    csv_manager.sleep_stage_csv_path = temp_csv_path
    
    # Set buffer size to maximum allowed
    csv_manager.sleep_stage_buffer_size = 1_000
    
    # Add many sleep stage entries
    for i in range(1_000):
        timestamp_start = i * 30.0
        timestamp_end = (i + 1) * 30.0
        csv_manager.add_sleep_stage_to_sleep_stage_csv(float(i % 5), float(i), timestamp_start, timestamp_end)
    
    # Save sleep stages
    result = csv_manager.save_sleep_stages_to_csv()
    assert result is True
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = pd.read_csv(temp_csv_path, delimiter='\t', header=None, skiprows=1,
                            names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
    
    assert len(saved_data) == 1_000
    for i in range(1_000):
        assert saved_data['sleep_stage'].iloc[i] == float(i % 5)
        assert saved_data['buffer_id'].iloc[i] == float(i)

def test_merge_files_successful(csv_manager, sample_data, temp_csv_path):
    """Test successful merge of main and sleep stage files.
    
    This test verifies that:
    1. The merge operation completes successfully
    2. Sleep stage and buffer ID columns are only populated for the end timestamp of each epoch
    3. Non-matching timestamps have NaN values for sleep stage and buffer ID
    4. The merged file contains all data from the main file
    5. Overlapping ranges don't cause overwriting - each sleep stage is only assigned to its end timestamp
    """
    # Create main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Create sleep stage CSV file with two epochs
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
    
    # Get timestamp channel index
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
    
    # Get timestamps from sample data
    timestamps = sample_data[timestamp_channel]
    
    # Create two epochs with some overlap
    # First epoch: 0.0 to 2.0
    first_start = timestamps[0]
    first_end = first_start + 2.0
    # Second epoch: 1.0 to 3.0 (overlaps with first epoch)
    second_start = first_start + 1.0
    second_end = first_start + 3.0
    
    # Add two sleep stage epochs
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, first_start, first_end)
    csv_manager.add_sleep_stage_to_sleep_stage_csv(3.0, 2.0, second_start, second_end)
    csv_manager.save_sleep_stages_to_csv()
    
    # Merge files
    output_path = temp_csv_path + '.merged'
    
    result = csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)
    assert result is True
    
    # Verify merged file exists
    assert os.path.exists(output_path)
    merged_data = pd.read_csv(output_path, delimiter='\t')
    
    # Verify all data from main file is preserved
    assert len(merged_data) == len(sample_data.T)
    
    # Verify sleep stage and buffer ID columns exist
    assert 'sleep_stage' in merged_data.columns
    assert 'buffer_id' in merged_data.columns
    
    # Find the exact end timestamps
    first_end_mask = merged_data['timestamp'] == first_end
    second_end_mask = merged_data['timestamp'] == second_end
    
    # Verify exactly two rows have sleep stage values (the end timestamps)
    assert merged_data.loc[first_end_mask, 'sleep_stage'].notna().sum() == 1, "First epoch end timestamp should have a sleep stage"
    assert merged_data.loc[second_end_mask, 'sleep_stage'].notna().sum() == 1, "Second epoch end timestamp should have a sleep stage"
    
    # Verify the correct values were assigned to the end timestamps
    assert merged_data.loc[first_end_mask, 'sleep_stage'].iloc[0] == 2.0, "First epoch end has wrong sleep stage"
    assert merged_data.loc[first_end_mask, 'buffer_id'].iloc[0] == 1.0, "First epoch end has wrong buffer ID"
    assert merged_data.loc[second_end_mask, 'sleep_stage'].iloc[0] == 3.0, "Second epoch end has wrong sleep stage"
    assert merged_data.loc[second_end_mask, 'buffer_id'].iloc[0] == 2.0, "Second epoch end has wrong buffer ID"
    
    # Verify all other timestamps have NaN values
    other_timestamps_mask = ~(first_end_mask | second_end_mask)
    assert merged_data.loc[other_timestamps_mask, 'sleep_stage'].isna().all(), "Non-end timestamps should have NaN sleep stage"
    assert merged_data.loc[other_timestamps_mask, 'buffer_id'].isna().all(), "Non-end timestamps should have NaN buffer ID"
    
    # Verify exactly 2 rows total have non-NaN values
    assert merged_data['sleep_stage'].notna().sum() == 2, "Should have exactly 2 rows with sleep stage values"
    assert merged_data['buffer_id'].notna().sum() == 2, "Should have exactly 2 rows with buffer ID values"

def test_merge_files_missing_sleep_stage_file(csv_manager, sample_data, temp_csv_path):
    """Test merge_files with missing sleep stage file.
    
    This test verifies that:
    1. The merge operation completes successfully even when sleep stage file is missing
    2. The merged file contains all data from the main file
    3. Sleep stage and buffer ID columns are added but contain only NaN values
    """
    # Create main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Try to merge with non-existent sleep stage file
    output_path = temp_csv_path + '.merged'
    result = csv_manager.merge_files(main_csv_path, 'nonexistent.sleep', output_path)
    
    # Verify merge was successful
    assert result is True
    
    # Read the merged file and verify contents
    merged_df = pd.read_csv(output_path, delimiter='\t')
    
    # Verify all original data is preserved
    assert len(merged_df) == len(sample_data.T)
    
    # Verify sleep stage and buffer ID columns exist and contain only NaN values
    assert 'sleep_stage' in merged_df.columns
    assert 'buffer_id' in merged_df.columns
    assert merged_df['sleep_stage'].isna().all()
    assert merged_df['buffer_id'].isna().all()

def test_merge_files_invalid_main_format(csv_manager, temp_csv_path):
    """Test merge_files with invalid main file format."""
    # Create invalid main CSV file with correct number of columns for BrainFlow data
    main_csv_path = temp_csv_path
    num_channels = BoardShim.get_num_rows(BoardIds.CYTON_DAISY_BOARD)
    
    with open(main_csv_path, 'w') as f:
        # Write invalid data with wrong number of columns
        f.write('\t'.join(['invalid'] * (num_channels - 1)) + '\n')

def test_merge_files_invalid_sleep_stage_format(csv_manager, sample_data, temp_csv_path):
    """Test merge_files with invalid sleep stage file format."""
    # Create main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Create invalid sleep stage CSV file
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    with open(sleep_stage_csv_path, 'w') as f:
        f.write('invalid,data\nmore,invalid\n')
    
    # Try to merge with invalid sleep stage file
    output_path = temp_csv_path + '.merged'
    with pytest.raises(CSVFormatError):
        csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)

def test_merge_files_empty_main_file(csv_manager, temp_csv_path):
    """Test merge_files with empty main file."""
    # Create empty main CSV file
    main_csv_path = temp_csv_path
    with open(main_csv_path, 'w') as f:
        f.write('')
    
    # Create sleep stage CSV file
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 100.0, 130.0)
    csv_manager.save_sleep_stages_to_csv()
    
    # Try to merge with empty main file
    output_path = temp_csv_path + '.merged'
    with pytest.raises(CSVDataError):
        csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)

def test_merge_files_empty_sleep_stage_file(csv_manager, sample_data, temp_csv_path):
    """Test merge_files with empty sleep stage file.
    
    This test verifies that:
    1. The merge operation completes successfully even when sleep stage file is empty
    2. The merged file contains all data from the main file
    3. Sleep stage and buffer ID columns are added but contain only NaN values
    """
    # Create main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Create empty sleep stage CSV file
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    with open(sleep_stage_csv_path, 'w') as f:
        f.write('timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n')
    
    # Merge with empty sleep stage file
    output_path = temp_csv_path + '.merged'
    result = csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)
    assert result is True
    
    # Read the merged file and verify contents
    merged_df = pd.read_csv(output_path, delimiter='\t')
    
    # Verify all original data is preserved
    assert len(merged_df) == len(sample_data.T)
    
    # Verify sleep stage and buffer ID columns exist and contain only NaN values
    assert 'sleep_stage' in merged_df.columns
    assert 'buffer_id' in merged_df.columns
    assert merged_df['sleep_stage'].isna().all()
    assert merged_df['buffer_id'].isna().all()

def test_merge_files_large_files(csv_manager, temp_csv_path):
    """Test merge_files with large files."""
    # Create large main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    
    # Set a larger buffer size for testing
    csv_manager.main_buffer_size = 100_000  # Increased from 10,000 to 100,000
    
    # Create data using create_brainflow_test_data for proper timestamp generation
    data, metadata = create_brainflow_test_data(
        duration_seconds=1000,  # 1000 seconds to get enough samples
        sampling_rate=100,      # 100 Hz to get 100,000 samples
        add_noise=False,        # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD
    )
    
    # Add data to buffer
    csv_manager.add_data_to_buffer(data.T, is_initial=True)  # Transpose to match expected shape
    csv_manager.save_main_buffer_to_csv()
    
    # Create sleep stage CSV file
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
    
    # Add sleep stages at regular intervals (every 100 seconds)
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
    for i in range(0, len(data), 10000):  # 10000 samples = 100 seconds at 100 Hz
        start_time = data[i, timestamp_channel]
        end_time = data[min(i + 10000, len(data) - 1), timestamp_channel]
        csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i // 10000, start_time, end_time)
    
    csv_manager.save_sleep_stages_to_csv()
    
    # Merge files
    output_path = temp_csv_path + '.merged'
    result = csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)
    assert result is True
    
    # Verify merged file exists and has correct content
    assert os.path.exists(output_path)
    merged_df = pd.read_csv(output_path, delimiter='\t')
    assert len(merged_df) == len(data)
    assert 'sleep_stage' in merged_df.columns
    assert 'buffer_id' in merged_df.columns

def test_csv_memory_management_full_flow():
    """
    Tests the complete flow of the CSV memory management system from data collection
    to final merged output, simulating a real recording session.
    """
    print("\n=== Starting test_csv_memory_management_full_flow ===")
    
    # 1. Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSVManager with small buffer sizes
        input_params = BrainFlowInputParams()
        csv_manager = CSVManager(
            board_shim=BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params),
            main_buffer_size=1000,  # Small buffer to test automatic saves
            sleep_stage_buffer_size=5  # Small buffer to test sleep stage saves
        )
        
        # Set up output paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        final_output_path = os.path.join(temp_dir, "test_data.merged.csv")
        
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
        
        print(f"Initial setup:")
        print(f"- Main CSV path: {main_csv_path}")
        print(f"- Sleep stage CSV path: {sleep_stage_csv_path}")
        print(f"- Sleep stage buffer size: {csv_manager.sleep_stage_buffer_size}")
        
        # 2. Simulate Recording Session
        # Generate 5 minutes of test data
        data, metadata = create_brainflow_test_data(
            duration_seconds=300,  # 5 minutes
            sampling_rate=125,     # 125 Hz
            add_noise=False,       # Clean data for easier verification
            board_id=BoardIds.CYTON_DAISY_BOARD
        )
        
        print(f"\nGenerated test data:")
        print(f"- Total samples: {len(data)}")
        print(f"- Start time: {metadata['start_time']}")
        
        # Define valid sleep stages
        SLEEP_STAGES = [0, 1, 2, 3, 4, 5]  # Wake, N1, N2, N3, REM, Unknown
        
        # Process data in chunks to simulate real-time collection
        chunk_size = 1000  # Process 1000 samples at a time
        last_sleep_stage_time = metadata['start_time']  # Track last sleep stage time
        timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            # Transpose chunk to match expected (channels, samples) shape
            chunk = chunk.T
            # Add to buffer - this will automatically save when buffer is full
            csv_manager.add_data_to_buffer(chunk, is_initial=(i == 0))
            
            # Get the latest timestamp from the data we've processed
            latest_processed_timestamp = data[i, timestamp_channel]
            
            # Add sleep stage if we've passed a 30-second interval
            while latest_processed_timestamp >= last_sleep_stage_time + 30:
                timestamp_start = last_sleep_stage_time
                # Find the closest timestamp in the data to the intended end time
                intended_end = timestamp_start + 30
                # Find the index where timestamp >= intended_end
                idx = np.searchsorted(data[:, timestamp_channel], intended_end, side='left')
                if idx >= len(data[:, timestamp_channel]):
                    idx = len(data[:, timestamp_channel]) - 1
                timestamp_end = str(data[idx, timestamp_channel])
                # Calculate sleep stage index based on time elapsed
                stage_index = int((timestamp_start - metadata['start_time']) / 30)
                sleep_stage = SLEEP_STAGES[stage_index % len(SLEEP_STAGES)]
                buffer_id = stage_index
                
                print(f"\nAdding sleep stage at time {timestamp_start}:")
                print(f"- Buffer ID: {buffer_id}")
                print(f"- Sleep stage: {sleep_stage}")
                print(f"- Timestamp range: {timestamp_start} to {timestamp_end}")
                print(f"- Current buffer size: {len(csv_manager.sleep_stage_buffer)}")
                
                csv_manager.add_sleep_stage_to_sleep_stage_csv(
                    sleep_stage=sleep_stage,
                    buffer_id=buffer_id,
                    timestamp_start=timestamp_start,
                    timestamp_end=float(timestamp_end)
                )
                
                print(f"After adding sleep stage:")
                print(f"- Buffer size: {len(csv_manager.sleep_stage_buffer)}")
                print(f"- File exists: {os.path.exists(sleep_stage_csv_path)}")
                if os.path.exists(sleep_stage_csv_path):
                    with open(sleep_stage_csv_path, 'r') as f:
                        content = f.read()
                        print(f"- File content:\n{content}")
                
                # Update last sleep stage time
                last_sleep_stage_time = float(timestamp_end)
        
        # Add final sleep stage if needed
        if last_sleep_stage_time < metadata['start_time'] + 300:  # 300 seconds = 5 minutes
            timestamp_start = last_sleep_stage_time
            intended_end = metadata['start_time'] + 300  # End at exactly 5 minutes
            idx = np.searchsorted(data[:, timestamp_channel], intended_end, side='left')
            if idx >= len(data[:, timestamp_channel]):
                idx = len(data[:, timestamp_channel]) - 1
            timestamp_end = str(data[idx, timestamp_channel])
            stage_index = int((timestamp_start - metadata['start_time']) / 30)
            sleep_stage = SLEEP_STAGES[stage_index % len(SLEEP_STAGES)]
            buffer_id = stage_index
            
            print(f"\nAdding final sleep stage at time {timestamp_start}:")
            print(f"- Buffer ID: {buffer_id}")
            print(f"- Sleep stage: {sleep_stage}")
            print(f"- Timestamp range: {timestamp_start} to {timestamp_end}")
            
            csv_manager.add_sleep_stage_to_sleep_stage_csv(
                sleep_stage=sleep_stage,
                buffer_id=buffer_id,
                timestamp_start=timestamp_start,
                timestamp_end=float(timestamp_end)
            )
        
        # 3. Save and Merge
        print("\nSaving remaining data...")
        csv_manager.save_all_and_cleanup()
        
        print("\nMerging files...")
        csv_manager.merge_files(
            main_csv_path=main_csv_path,
            sleep_stage_csv_path=sleep_stage_csv_path,
            output_path=final_output_path
        )
        
        # 4. Verify Results
        print("\nVerifying results...")
        # Check main CSV file
        saved_data = np.loadtxt(main_csv_path, delimiter='\t')
        print(f"Main CSV:")
        print(f"- Saved samples: {len(saved_data)}")
        print(f"- Expected samples: {len(data)}")
        assert len(saved_data) == len(data), "Not all samples were saved"
        assert np.allclose(saved_data, data, rtol=1e-5, atol=1e-5), "Data integrity check failed"
        
        # Check sleep stage CSV file
        sleep_stage_data = pd.read_csv(sleep_stage_csv_path, delimiter='\t', header=None, skiprows=1,
                                     names=['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id'])
        expected_num_stages = 300 // 30
        print(f"\nSleep stage CSV:")
        print(f"- Saved stages: {len(sleep_stage_data)}")
        print(f"- Expected stages: {expected_num_stages}")
        print(f"- Content:\n{sleep_stage_data}")
        assert len(sleep_stage_data) == expected_num_stages, "Not all sleep stages were saved"
        
        # Convert timestamps to strings for exact matching
        sleep_stage_data['timestamp_end'] = sleep_stage_data['timestamp_end'].astype(str)
        
        # Build set of actual sleep stage end timestamps from the sleep stage CSV
        all_epoch_ends = set(sleep_stage_data['timestamp_end'])
        
        # Verify sleep stages cycle correctly
        for i in range(expected_num_stages):
            expected_stage = SLEEP_STAGES[i % len(SLEEP_STAGES)]
            assert sleep_stage_data['sleep_stage'].iloc[i] == expected_stage, f"Sleep stage mismatch at index {i}"
        
        # Check final merged file
        merged_data = pd.read_csv(final_output_path, delimiter='\t')
        print(f"\nMerged file:")
        print(f"- Total rows: {len(merged_data)}")
        print(f"- Columns: {merged_data.columns.tolist()}")
        assert len(merged_data) == len(data), "Merged file missing data points"
        assert 'sleep_stage' in merged_data.columns, "Sleep stage column missing from merged file"
        assert 'buffer_id' in merged_data.columns, "Buffer ID column missing from merged file"
        
        # Convert timestamps to strings for exact matching
        merged_data['timestamp'] = merged_data['timestamp'].astype(str)
        
        # Verify sleep stage alignment
        for i in range(expected_num_stages):
            # Get the actual end timestamp for this epoch from the sleep stage CSV
            end_time = sleep_stage_data['timestamp_end'].iloc[i]
            
            # Find the exact end timestamp row
            end_row = merged_data[merged_data['timestamp'] == end_time]
            
            # Log expected vs actual sleep stage data
            print(f"\nEpoch {i} Analysis:")
            print(f"Expected end timestamp: {end_time}")
            
            # Find all rows in this epoch that have sleep stage data
            epoch_data = merged_data[
                (merged_data['timestamp'] > str(sleep_stage_data['timestamp_start'].iloc[i])) & 
                (merged_data['timestamp'] < end_time)
            ]
            
            # Find rows that have sleep stage data but are not ends of any epoch
            non_end_rows_with_data = epoch_data[
                (epoch_data['sleep_stage'].notna()) &
                (~epoch_data['timestamp'].isin(all_epoch_ends))
            ]
            
            if not non_end_rows_with_data.empty:
                print(f"Found {len(non_end_rows_with_data)} rows with sleep stage data when they should be NaN:")
                for idx, row in non_end_rows_with_data.iterrows():
                    print(f"  Timestamp: {row['timestamp']}, Sleep Stage: {row['sleep_stage']}, Buffer ID: {row['buffer_id']}")
            
            # Verify sleep stage and buffer ID at end timestamp
            if not end_row.empty:
                expected_stage = SLEEP_STAGES[i % len(SLEEP_STAGES)]
                print(f"End timestamp {end_time} has sleep stage: {end_row['sleep_stage'].iloc[0]}")
                assert end_row['sleep_stage'].iloc[0] == expected_stage, \
                    f"Sleep stage {expected_stage} not found at end timestamp {end_time}"
                assert end_row['buffer_id'].iloc[0] == i, \
                    f"Buffer ID {i} not found at end timestamp {end_time}"
            else:
                print(f"Warning: No row found for end timestamp {end_time}")
            
            # Verify that only timestamps that are ends of epochs have sleep stage values
            # Note: A timestamp can be the end of a previous epoch while also being part of the current epoch
            # So we need to check if the timestamp is the end of ANY epoch, not just the current one
            rows_with_data = epoch_data[epoch_data['sleep_stage'].notna()]
            for _, row in rows_with_data.iterrows():
                timestamp = str(row['timestamp'])
                assert timestamp in all_epoch_ends, \
                    f"Found timestamp {timestamp} with sleep stage value when it should be NaN in epoch {i}"
        
        print("\n=== Test completed successfully ===\n")

def test_buffer_management_add_then_check(csv_manager, temp_csv_path):
    """Test that data is added first, then buffer size is checked."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Set a small buffer size for testing
    csv_manager.main_buffer_size = 5
    
    # Create test data using create_brainflow_test_data
    # For 0.1 seconds at 125 Hz: 12.5 samples (rounded to 12)
    data, metadata = create_brainflow_test_data(
        duration_seconds=0.1,  # Short duration for small test
        sampling_rate=125,     # 125 Hz (Cyton Daisy standard)
        add_noise=False,       # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD,
        start_time=1700000000.1  # Explicit start time
    )
    
    # Add data to buffer
    result = csv_manager.add_data_to_buffer(data.T, is_initial=True)  # Transpose to match expected shape
    assert result is True
    
    # Verify data was saved to CSV (not in buffer) since it exceeds buffer size
    assert len(csv_manager.main_csv_buffer) == 0  # Buffer should be empty
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) == 12  # 0.1 seconds * 125 Hz = 12.5 samples (rounded to 12)
    
    # Create more test data that would exceed buffer size
    # For 0.1 seconds at 125 Hz: 12.5 samples (rounded to 12)
    # Start time should be after the first chunk's end time
    subsequent_start_time = 1700000000.1 + 0.1  # Initial start time + initial duration
    more_data, _ = create_brainflow_test_data(
        duration_seconds=0.1,  # Short duration for small test
        sampling_rate=125,     # 125 Hz (Cyton Daisy standard)
        add_noise=False,       # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD,
        start_time=subsequent_start_time  # Start after first chunk ends
    )
    
    # This should trigger a save since total size would exceed buffer_size
    result = csv_manager.add_data_to_buffer(more_data.T)  # Transpose to match expected shape
    assert result is True
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) == 24  # 12 samples from first save + 12 samples from second save
    
    # Verify buffer is empty after save
    assert len(csv_manager.main_csv_buffer) == 0

def test_buffer_management_large_initial_data(csv_manager, temp_csv_path):
    """Test handling of large initial data chunk."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Set buffer size
    csv_manager.main_buffer_size = 5
    
    # Create large initial data using create_brainflow_test_data
    # For 0.2 seconds at 125 Hz: 25 samples
    data, metadata = create_brainflow_test_data(
        duration_seconds=0.2,  # Longer duration for larger test
        sampling_rate=125,     # 125 Hz (Cyton Daisy standard)
        add_noise=False,       # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD
    )
    
    # Add large initial data
    result = csv_manager.add_data_to_buffer(data.T, is_initial=True)  # Transpose to match expected shape
    assert result is True
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) == 25  # 0.2 seconds * 125 Hz = 25 samples
    
    # Verify buffer is empty
    assert len(csv_manager.main_csv_buffer) == 0

def test_buffer_management_subsequent_data(csv_manager, temp_csv_path):
    """Test handling of subsequent data chunks."""
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Set buffer size
    csv_manager.main_buffer_size = 5
    
    # Add initial data using create_brainflow_test_data
    # For 0.1 seconds at 125 Hz: 12.5 samples (rounded to 12)
    initial_data, metadata = create_brainflow_test_data(
        duration_seconds=0.1,  # Short duration for small test
        sampling_rate=125,     # 125 Hz (Cyton Daisy standard)
        add_noise=False,       # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD,
        start_time=1700000000.1  # Explicit start time
    )
    csv_manager.add_data_to_buffer(initial_data.T, is_initial=True)  # Transpose to match expected shape
    
    # Add subsequent data that would exceed buffer size
    # For 0.2 seconds at 125 Hz: 25 samples
    # Start time should be after the initial data's last timestamp
    subsequent_start_time = 1700000000.1 + 0.1  # Initial start time + initial duration
    subsequent_data, _ = create_brainflow_test_data(
        duration_seconds=0.2,  # Longer duration for larger test
        sampling_rate=125,     # 125 Hz (Cyton Daisy standard)
        add_noise=False,       # Clean data for easier verification
        board_id=BoardIds.CYTON_DAISY_BOARD,
        start_time=subsequent_start_time  # Start after initial data ends
    )
    
    # This should trigger a save since total size would exceed buffer_size
    result = csv_manager.add_data_to_buffer(subsequent_data.T)  # Transpose to match expected shape
    assert result is True
    
    # Verify file exists and has correct content
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) == 37  # 12 samples from first save + 25 samples from second save
    
    # Verify buffer is empty after save
    assert len(csv_manager.main_csv_buffer) == 0

def test_merge_files_no_sleep_stage_file(csv_manager, sample_data, temp_csv_path):
    """Test merge_files when sleep stage file doesn't exist.
    
    This test verifies that:
    1. The merge operation completes successfully even when sleep stage file is missing
    2. The merged file contains all data from the main file
    3. Sleep stage and buffer ID columns are added but contain only NaN values
    """
    # Create main CSV file
    main_csv_path = temp_csv_path
    csv_manager.main_csv_path = main_csv_path
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Create a non-existent sleep stage file path
    sleep_stage_csv_path = temp_csv_path + '.sleep'
    
    # Merge files
    output_path = temp_csv_path + '.merged'
    result = csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, output_path)
    assert result is True
    
    # Verify merged file exists
    assert os.path.exists(output_path)
    merged_data = pd.read_csv(output_path, delimiter='\t')
    
    # Verify all data from main file is preserved
    assert len(merged_data) == len(sample_data.T)
    
    # Verify sleep stage and buffer ID columns exist
    assert 'sleep_stage' in merged_data.columns
    assert 'buffer_id' in merged_data.columns
    
    # Verify all values in sleep stage and buffer ID columns are NaN
    assert merged_data['sleep_stage'].isna().all(), "Sleep stage column should contain only NaN values"
    assert merged_data['buffer_id'].isna().all(), "Buffer ID column should contain only NaN values"

def test_save_all_data_with_empty_buffers(csv_manager, sample_data, temp_csv_path):
    """Test that save_all_data() doesn't raise an error when buffers are empty but data exists in CSV."""
    # Set both paths
    csv_manager.main_csv_path = temp_csv_path
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    
    # Add data to buffer and save it incrementally
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    csv_manager.save_main_buffer_to_csv()
    
    # Verify buffer is empty after save
    assert len(csv_manager.main_csv_buffer) == 0
    
    # Verify data exists in CSV file
    assert os.path.exists(temp_csv_path)
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) > 0
    
    # Call save_all_data() with empty buffers
    result = csv_manager.save_all_data()
    assert result is True  # Should succeed even with empty buffers

def test_save_new_data_initial(csv_manager, sample_data):
    """Test saving initial data."""
    result = csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    assert result is True
    assert len(csv_manager.main_csv_buffer) == sample_data.shape[1]
    # Get timestamp from the timestamp channel
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
    assert csv_manager.last_saved_timestamp == sample_data[timestamp_channel, -1]

def test_save_new_data_subsequent(csv_manager, sample_data):
    """Test saving subsequent data."""
    # Save initial data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    initial_length = len(csv_manager.main_csv_buffer)
    
    # Create new data with some overlap (last 5 samples)
    new_data = sample_data[:, -5:]
    result = csv_manager.add_data_to_buffer(new_data)
    assert result is True
    
    # Buffer length should not change since all timestamps are duplicates
    assert len(csv_manager.main_csv_buffer) == initial_length
    
    # Now create new data with timestamps after the initial data
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
    last_timestamp = sample_data[timestamp_channel, -1]
    
    # Create new data with timestamps after the last timestamp
    duration_seconds = 0.1  # 0.1 seconds at 125 Hz = ~12 samples
    new_data, _ = create_brainflow_test_data(
        duration_seconds=duration_seconds,
        sampling_rate=125,
        add_noise=False,
        board_id=BoardIds.CYTON_DAISY_BOARD,
        start_time=last_timestamp + 0.1  # Start 0.1 seconds after last timestamp
    )
    
    # Add the new data
    result = csv_manager.add_data_to_buffer(new_data.T)
    assert result is True
    
    # Buffer length should increase since timestamps are new
    assert len(csv_manager.main_csv_buffer) > initial_length

def test_backward_compatibility_save_all_and_cleanup(csv_manager, sample_data, temp_csv_path):
    """Test the save_all_and_cleanup convenience method.
    
    Note: save_all_and_cleanup() is not a deprecated method, but a new convenience method
    that combines save_all_data() and cleanup(). This test ensures it works correctly.
    """
    # Set main path
    csv_manager.main_csv_path = temp_csv_path
    
    # Save some data
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)
    
    # Add sleep stage data with new method signature
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, 0.0, 30.0)
    
    # Test that the method works correctly
    result = csv_manager.save_all_and_cleanup()
    assert result is True
    
    # Verify cleanup was performed
    assert len(csv_manager.main_csv_buffer) == 0
    assert len(csv_manager.sleep_stage_buffer) == 0
    assert csv_manager.last_saved_timestamp is None
    assert csv_manager.main_csv_path is None
    assert csv_manager.sleep_stage_csv_path is None

def test_save_all_data(csv_manager, sample_data, temp_csv_path):
    """Test saving all data without cleanup.
    Run pytest with -s to see debug output.
    """
    # Add some data to the main buffer
    csv_manager.add_data_to_buffer(sample_data, is_initial=True)

    # Add some sleep stage data
    csv_manager.sleep_stage_csv_path = temp_csv_path + '.sleep'
    timestamp_start = 100.0
    timestamp_end = 130.0
    csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, 1.0, timestamp_start, timestamp_end)

    # Set the main path
    csv_manager.main_csv_path = temp_csv_path

    # Call save_all_data
    result = csv_manager.save_all_data()
    assert result is True

    # Verify files exist and have correct content
    assert os.path.exists(temp_csv_path)
    assert os.path.exists(temp_csv_path + '.sleep')

    # Verify main CSV content
    saved_data = np.loadtxt(temp_csv_path, delimiter='\t')
    assert len(saved_data) > 0

    # Verify sleep stage CSV content using pandas
    sleep_stage_data = pd.read_csv(temp_csv_path + '.sleep', delimiter='\t')
    assert len(sleep_stage_data) == 1
    assert sleep_stage_data['sleep_stage'].iloc[0] == 2.0
    assert sleep_stage_data['buffer_id'].iloc[0] == 1.0
    assert sleep_stage_data['timestamp_start'].iloc[0] == timestamp_start
    assert sleep_stage_data['timestamp_end'].iloc[0] == timestamp_end

if __name__ == '__main__':
    print("\nRunning tests directly...")
    pytest.main([__file__, '-v'])

# python -m pytest gssc_local/realtime_with_restart/export/test_csv_manager.py -v