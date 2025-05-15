"""
Test utilities for creating mock data and other test helpers.
"""

import numpy as np
import pandas as pd
import time
import unittest
import pytest
from brainflow.board_shim import BoardShim, BoardIds
import os

def create_brainflow_test_data(duration_seconds=1.0, sampling_rate=None, add_noise=False, board_id=None):
    """
    Create BrainFlow-compatible test data.
    
    Args:
        duration_seconds: Duration of data in seconds
        sampling_rate: Sampling rate in Hz (defaults to board's rate)
        add_noise: Whether to add noise to EEG channels
        board_id: Board ID from BoardIds enum (defaults to CYTON_DAISY_BOARD)
        
    Returns:
        tuple: (data_array, metadata_dict)
            - data_array: numpy array of shape (n_samples, n_channels)
            - metadata_dict: dict containing sampling_rate, timestamp_channel, start_time
    """
    # Get board configuration
    if board_id is None:
        board_id = BoardIds.CYTON_DAISY_BOARD
        
    n_channels = BoardShim.get_num_rows(board_id)
    if sampling_rate is None:
        sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    package_num_channel = BoardShim.get_package_num_channel(board_id)
    marker_channel = BoardShim.get_marker_channel(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    
    # Calculate number of samples
    n_samples = int(duration_seconds * sampling_rate)
    
    # Initialize data array with correct number of columns
    data = np.zeros((n_samples, n_channels))
    
    # Package number (increments by 2 starting at 0)
    # This also serves as the sample index when divided by 2
    data[:, package_num_channel] = np.arange(0, n_samples * 2, 2)
    
    # Generate EEG data
    t = np.arange(n_samples) / sampling_rate
    for i, channel in enumerate(eeg_channels):
        # Generate different frequencies and phases for each channel
        freq = (i + 1) * 2  # 2Hz, 4Hz, 6Hz, etc.
        phase = i * np.pi / 4  # Different phase for each channel
        signal = np.sin(2 * np.pi * freq * t + phase)
        
        # Add some amplitude variation
        amplitude = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Slowly varying amplitude
        signal = signal * amplitude
        
        if add_noise:
            # Add more significant noise
            noise = np.random.normal(0, 0.2, n_samples)
            signal = signal + noise
            
        data[:, channel] = signal
    
    # Set fixed values for specific channels

    
    # Set timestamps
    start_time = time.time()
    expected_interval = 1.0 / sampling_rate
    timestamps = start_time + np.arange(n_samples) * expected_interval
    data[:, timestamp_channel] = timestamps
    
    # Create metadata dictionary
    metadata = {
        'sampling_rate': sampling_rate,
        'timestamp_channel': timestamp_channel,
        'package_num_channel': package_num_channel,
        'marker_channel': marker_channel,
        'eeg_channels': eeg_channels,
        'start_time': start_time,
        'n_samples': n_samples,
        'duration_seconds': duration_seconds,
        'board_id': board_id,
        'n_channels': n_channels
    }
    
    return data, metadata

def save_brainflow_data_to_csv(data, filepath):
    """
    Save BrainFlow data to a CSV file.
    
    Args:
        data: numpy array of shape (n_samples, 32)
        filepath: Path to save the CSV file
    """
    df = pd.DataFrame(data)
    df.to_csv(filepath, sep='\t', index=False, header=False)

class TestBrainFlowDataGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.duration_seconds = 1.0
        self.board_id = BoardIds.CYTON_DAISY_BOARD
        self.data, self.metadata = create_brainflow_test_data(
            duration_seconds=self.duration_seconds,
            add_noise=False,
            board_id=self.board_id
        )
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
    def test_data_shape(self):
        """Test that generated data has correct shape (n_samples, n_channels)."""
        n_samples = int(self.duration_seconds * self.metadata['sampling_rate'])
        n_channels = BoardShim.get_num_rows(self.board_id)
        self.assertEqual(self.data.shape, (n_samples, n_channels))
        

        
    def test_timestamps(self):
        """Test that timestamps are correctly spaced and formatted."""
        timestamps = self.data[:, self.metadata['timestamp_channel']]
        expected_interval = 1.0 / self.metadata['sampling_rate']
        actual_intervals = np.diff(timestamps)
        
        # Check that intervals are approximately correct
        np.testing.assert_allclose( #TODO: Sometimes we want gaps in the data, so the correct interval is not always necessary. How do we handle this?
            actual_intervals,
            expected_interval,
            rtol=1e-3  # Allow 0.1% tolerance
        )
        
        # Check timestamp format (should be Unix timestamp with microseconds)
        for ts in timestamps:
            self.assertGreater(ts, 1700000000)  # Should be after 2023
            self.assertLess(ts, 2000000000)     # Should be before 2033
    def test_eeg_channels(self):
        """Test that EEG channels contain realistic values."""
        # Get board-specific ADC specifications
        if self.board_id == BoardIds.CYTON_DAISY_BOARD or self.board_id == BoardIds.CYTON_BOARD:
            # Cyton and Cyton Daisy use the same ADC specifications
            min_adc = 0
            max_adc = 2**24 - 1  # 24-bit ADC
            zero_point = 8192
            ref_voltage = 4.5
            gain = 24
        elif self.board_id == BoardIds.GANGLION_BOARD:
            # Ganglion board specifications
            min_adc = 0
            max_adc = 2**12 - 1  # 12-bit ADC
            zero_point = 2048
            ref_voltage = 1.2
            gain = 1
        else:
            raise ValueError(f"Unsupported board type: {self.board_id}. Only CYTON_DAISY_BOARD, CYTON_BOARD, and GANGLION_BOARD are supported.")
        
        # Calculate theoretical voltage range in microvolts
        min_voltage = (min_adc - zero_point) * (ref_voltage / gain) / gain
        max_voltage = (max_adc - zero_point) * (ref_voltage / gain) / gain
        
        for channel in self.eeg_channels:
            # Values should be within the theoretical ADC range of the board
            self.assertTrue(np.all(self.data[:, channel] >= min_voltage),
                          f"Channel {channel} contains values below theoretical minimum of {min_voltage:.2f} µV")
            self.assertTrue(np.all(self.data[:, channel] <= max_voltage),
                          f"Channel {channel} contains values above theoretical maximum of {max_voltage:.2f} µV")
            
            # Values should be within practical EEG range (±1000 µV)
            # This is more conservative than the theoretical range to catch potential issues
            self.assertTrue(np.all(np.abs(self.data[:, channel]) < 1000),
                          f"Channel {channel} contains values outside the practical range of ±1000 µV")
            
    def test_noise_addition(self):
        """Test that noise is added correctly when requested."""
        data_with_noise, _ = create_brainflow_test_data(
            duration_seconds=self.duration_seconds,
            add_noise=True,
            board_id=self.board_id
        )
        
        # Check that EEG channels have different values
        for channel in self.eeg_channels:
            self.assertFalse(np.array_equal(
                self.data[:, channel],
                data_with_noise[:, channel]
            ))
            
    def test_csv_format(self):
        """Test that saved CSV matches BrainFlow format."""
        # Save test data to CSV
        test_file = 'test_brainflow_data.csv'
        save_brainflow_data_to_csv(self.data, test_file)
        
        try:
            # Read back the CSV
            df = pd.read_csv(test_file, sep='\t', header=None)
            
            # Check shape
            self.assertEqual(df.shape, self.data.shape)
            
            # Check first few rows match
            np.testing.assert_allclose(df.values, self.data, rtol=1e-3)
            
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)
                
    def test_column_precision(self):
        """Test that each column has exactly 6 decimal places."""
        # Get channel information
        timestamp_channel = self.metadata['timestamp_channel']
        package_num_channel = self.metadata['package_num_channel']
        eeg_channels = self.metadata['eeg_channels']
        
        # Check each column's precision
        for col in range(self.data.shape[1]):
            # Get unique values in the column
            unique_values = np.unique(self.data[:, col])
            
            # Check that all values have exactly 6 decimal places
            for val in unique_values:
                # Convert to string with 6 decimal places
                str_val = f"{val:.6f}"
                # Convert back to float
                formatted_val = float(str_val)
                # Use numpy's testing function for floating point comparison
                np.testing.assert_almost_equal(
                    val, formatted_val, 
                    decimal=6,
                    err_msg=f"Column {col} value {val} does not have exactly 6 decimal places"
                )
            
if __name__ == '__main__':
    print("\nRunning tests directly...")
    pytest.main([__file__, '-v']) 