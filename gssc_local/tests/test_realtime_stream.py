#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import multiprocessing
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
import time
import logging
from ..realtime_with_restart.export.csv.utils import MAIN_DATA_FMT
from ..realtime_with_restart.export.csv.manager import CSVManager
from ..realtime_with_restart.core.brainflow_child_process_manager import BrainFlowChildProcessManager
from ..realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import the components we'll be testing
from gssc_local.realtime_with_restart.main_multiprocess import main
from gssc_local.realtime_with_restart.data_manager import DataManager
from gssc_local.realtime_with_restart.board_manager import BoardManager

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(processName)s - %(levelname)s - L%(lineno)s - %(message)s'
)

class TestRealtimeStream(unittest.TestCase):
    def setUp(self):
        """Setup test fixtures before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data_dir = os.path.join(self.temp_dir.name, 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create mock data file
        self.mock_data_file = self._create_mock_data_file()
        
        # Setup mocks
        self._setup_mocks()
        
    def tearDown(self):
        """Cleanup after each test."""
        self.temp_dir.cleanup()

    def _create_mock_data_file(self):
        """Create a mock CSV file with dummy data for testing.
        
        Creates a BrainFlow-compatible CSV file with:
        - 10 seconds of normal data
        - 2 second gap
        - 10 seconds of data after gap
        """
        # Get board configuration
        master_board_id = BoardIds.CYTON_DAISY_BOARD
        sampling_rate = BoardShim.get_sampling_rate(master_board_id)  # 125 Hz
        timestamp_channel = BoardShim.get_timestamp_channel(master_board_id)
        
        # Calculate samples for each segment
        samples_before_gap = int(10 * sampling_rate)  # 10 seconds
        samples_after_gap = int(10 * sampling_rate)   # 10 seconds
        total_samples = samples_before_gap + samples_after_gap
        
        # Initialize data array with 32 columns (BrainFlow format)
        data = np.zeros((total_samples, 32))
        
        # Generate timestamps with a gap
        start_time = time.time()
        expected_interval = 1.0 / sampling_rate
        
        # Timestamps for first segment
        timestamps_before = start_time + np.arange(samples_before_gap) * expected_interval
        
        # Timestamps for second segment (after gap)
        gap_duration = 2.0  # 2 second gap
        timestamps_after = (timestamps_before[-1] + gap_duration + 
                          np.arange(1, samples_after_gap + 1) * expected_interval)
        
        # Combine timestamps
        timestamps = np.concatenate([timestamps_before, timestamps_after])
        data[:, timestamp_channel] = timestamps
        
        # Fill in other columns with test data
        # Column 1: Package number
        data[:, 0] = np.arange(total_samples) + 2
        
        # Column 2: Sample index
        data[:, 1] = np.arange(1, total_samples + 1)
        
        # Columns 3-17: Consecutive numbers
        base_value = 2916
        for i in range(2, 17):
            data[:, i] = base_value + np.arange(total_samples)
        
        # Columns 18-20: EEG data (sine waves for easy visualization)
        t = np.arange(total_samples) / sampling_rate
        for i in range(17, 20):
            data[:, i] = np.sin(2 * np.pi * (i-16) * t)  # Different frequencies for each channel
        
        # Column 21: Fixed value
        data[:, 20] = 192.0
        
        # Save to CSV in BrainFlow format
        file_path = os.path.join(self.test_data_dir, 'mock_data_with_gap.csv')
        
        # Create format string for each column (32 columns total)
        fmt = '\t'.join([MAIN_DATA_FMT] * 32)
        
        np.savetxt(
            file_path,
            data,
            delimiter='\t',
            fmt=fmt
        )
        
        # Store metadata for assertions
        self.gap_start_sample = samples_before_gap
        self.gap_duration = gap_duration
        self.sampling_rate = sampling_rate
        self.timestamp_channel = timestamp_channel
        self.start_time = start_time
        
        print(f"\nCreated mock data file: {file_path}")
        print(f"Total samples: {total_samples}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Gap duration: {gap_duration} seconds")
        print(f"Gap starts at sample: {self.gap_start_sample}")
        
        return file_path

    def test_mock_data_generation(self):
        """Test that the mock data file is generated correctly with the expected gap."""
        print("\nRunning test_mock_data_generation...")
        
        # Load the generated data
        data = np.loadtxt(self.mock_data_file, delimiter='\t')
        print(f"Loaded data shape: {data.shape}")
        
        # Test 1: Check file exists and has correct shape
        self.assertTrue(os.path.exists(self.mock_data_file))
        self.assertEqual(data.shape[1], 32)  # 32 columns
        print("✓ File exists and has correct number of columns")
        
        # Test 2: Verify sampling rate and total duration
        expected_samples = int(20 * self.sampling_rate)  # 20 seconds total (10s + 10s, gap doesn't count)
        self.assertEqual(data.shape[0], expected_samples)
        print(f"✓ Has correct number of samples: {expected_samples}")
        
        # Test 3: Check timestamp gap
        timestamps = data[:, self.timestamp_channel]
        gap_start_idx = self.gap_start_sample - 1  # Last sample before gap
        gap_end_idx = self.gap_start_sample  # First sample after gap
        
        # Calculate time difference at gap
        time_diff = timestamps[gap_end_idx] - timestamps[gap_start_idx]
        self.assertAlmostEqual(time_diff, self.gap_duration + 1/self.sampling_rate, places=2)
        print(f"✓ Gap duration is correct: {time_diff:.2f} seconds")
        
        # Test 4: Verify EEG data format (sine waves)
        for ch in range(17, 20):  # EEG channels
            channel_data = data[:, ch]
            # Check that data is bounded between -1 and 1 (sine wave)
            self.assertTrue(np.all(channel_data >= -1) and np.all(channel_data <= 1))
            # Check that it's not all zeros
            self.assertFalse(np.all(channel_data == 0))
        print("✓ EEG channels contain valid sine waves")
        
        # Test 5: Verify fixed value column
        self.assertTrue(np.all(data[:, 20] == 192.0))
        print("✓ Fixed value column is correct")
        
        # Test 6: Check consecutive numbers in package and sample columns
        self.assertTrue(np.all(np.diff(data[:, 0]) == 1))  # Package numbers
        self.assertTrue(np.all(np.diff(data[:, 1]) == 1))  # Sample indices
        print("✓ Package and sample numbers are consecutive")
        
        # Test 7: Verify base value in consecutive number columns
        for i in range(2, 17):
            self.assertTrue(np.all(data[:, i] >= 2916))  # Base value check
        print("✓ Base values are correct in all columns")
        
        print("\nAll tests passed! ✨")

    def _setup_mocks(self):
        """Setup all necessary mocks for the test."""
        # Mock BoardManager
        self.mock_board_manager = Mock(spec=BoardManager)
        self.mock_board_manager.board_shim = Mock()
        self.mock_board_manager.sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)  # Get actual sampling rate
        self.mock_board_manager.board_timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)  # Get actual timestamp channel
        
        # Mock BoardShim methods
        self.mock_board_manager.board_shim.get_board_id.return_value = BoardIds.CYTON_DAISY_BOARD
        self.mock_board_manager.board_shim.get_exg_channels.return_value = list(range(8))  # 8 EEG channels
        self.mock_board_manager.board_shim.get_timestamp_channel.return_value = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)
        
        # Mock Qt App
        self.mock_qt_app = Mock()
        self.mock_qt_app.processEvents = Mock()
        
        # Mock Visualizer
        self.mock_visualizer = Mock()
        self.mock_visualizer.app = self.mock_qt_app
        
        # Mock DataManager
        self.mock_data_manager = Mock(spec=DataManager)
        self.mock_data_manager.queue_data_for_csv_write = Mock()
        self.mock_data_manager.add_to_data_processing_buffer.return_value = True
        self.mock_data_manager.save_new_data = Mock()
        self.mock_data_manager._calculate_next_buffer_id_to_process.return_value = 0
        self.mock_data_manager.next_available_epoch_on_round_robin_buffer.return_value = (True, None, 0, 1250)  # 10 seconds at 125 Hz
        self.mock_data_manager.manage_epoch.return_value = [1]  # Mock sleep stage prediction
        self.mock_data_manager.add_sleep_stage_to_csv = Mock()
        self.mock_data_manager.visualizer = self.mock_visualizer
        
        # Mock ReceivedStreamedDataHandler
        self.mock_data_handler = Mock(spec=ReceivedStreamedDataHandler)
        self.mock_data_handler.process_board_data = Mock()
        self.mock_data_handler.data_manager = self.mock_data_manager
        
        # Mock logger
        self.mock_logger = Mock()
        self.mock_logger.info = Mock()
        self.mock_logger.error = Mock()
        self.mock_logger.warning = Mock()
        
        # Store mocks as instance variables for test assertions
        self.mocks = {
            'board_manager': self.mock_board_manager,
            'data_manager': self.mock_data_manager,
            'data_handler': self.mock_data_handler,
            'logger': self.mock_logger,
            'qt_app': self.mock_qt_app,
            'visualizer': self.mock_visualizer
        }
        
        print("\nMocks setup complete:")
        print("✓ BoardManager mocked with CYTON_DAISY_BOARD configuration")
        print("✓ DataManager mocked with basic processing methods")
        print("✓ ReceivedStreamedDataHandler mocked with process_board_data")
        print("✓ Logger mocked with info/error/warning methods")
        print("✓ Qt App and Visualizer mocked")

    @patch('gssc_local.realtime_with_restart.main_multiprocess.os.path.isfile')
    @patch('gssc_local.realtime_with_restart.main_multiprocess.create_trimmed_csv')
    @patch('gssc_local.realtime_with_restart.main_multiprocess.BoardManager')
    @patch('gssc_local.realtime_with_restart.main_multiprocess.BrainFlowChildProcessManager')
    @patch('gssc_local.realtime_with_restart.main_multiprocess.logger')
    def test_stream_restart_and_trim_flow(self, mock_logger, mock_stream_manager_class, 
                                        mock_board_manager_class, mock_create_trimmed, mock_isfile):
        """Test the complete flow of streaming, gap detection, and restart."""
        print("\nTesting stream restart and trim flow...")
        
        # Setup mock file operations
        mock_isfile.return_value = True
        mock_create_trimmed.return_value = None
        
        # Setup mock stream manager
        mock_stream_manager = Mock(spec=BrainFlowChildProcessManager)
        mock_stream_manager_class.return_value = mock_stream_manager
        
        # Setup message sequence
        message_sequence = [
            ('start_ts', None),  # Initial timestamp
            ('data', {'board_data': np.zeros((32, 100))}),  # First data chunk
            ('data', {'board_data': np.zeros((32, 100))}),  # Second data chunk
            ('last_ts', time.time())  # Gap detection
        ]
        
        def mock_get_next_message():
            if message_sequence:
                msg = message_sequence.pop(0)
                print(f"mock_get_next_message: returning {msg}")
                return msg
            print("mock_get_next_message: returning None")
            return None
        
        mock_stream_manager.get_next_message.side_effect = mock_get_next_message
        mock_stream_manager.is_streaming.side_effect = [True, True, True, True, True, False]
        
        # Setup mock board manager
        mock_board_manager_class.return_value = self.mock_board_manager
        
        # Create a mock handler class and instance
        mock_handler = Mock(spec=ReceivedStreamedDataHandler)
        process_calls = []
        def track_process_calls(*args, **kwargs):
            process_calls.append((args, kwargs))
        mock_handler.process_board_data.side_effect = track_process_calls
        mock_handler.data_manager = self.mock_data_manager
        mock_handler_class = Mock(return_value=mock_handler)
        
        # Mock DataFrame for file reading
        mock_df = pd.DataFrame(np.zeros((1000, 32)))
        mock_df.iloc[:, self.mock_board_manager.board_timestamp_channel] = np.arange(1000) / 125.0  # timestamps
        
        # Run the pipeline
        with patch('gssc_local.realtime_with_restart.main_multiprocess.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = mock_df
            main(handler_class=mock_handler_class)
        print(f"process_board_data call count: {mock_handler.process_board_data.call_count}")
        
        # Verify stream manager was used correctly
        mock_stream_manager_class.assert_called_once()
        mock_stream_manager.start_stream.assert_called_once()
        mock_stream_manager.stop_stream.assert_called_once()
        
        # Verify data was processed using the mock instance that was created
        mock_handler_class.assert_called_once()  # Verify the class was instantiated
        self.assertTrue(len(process_calls) > 0, "Expected process_board_data to have been called")
        
        # Verify the number of data processing calls
        expected_calls = 2  # We sent 2 data chunks
        actual_calls = len(process_calls)
        self.assertEqual(actual_calls, expected_calls, 
                        f"Expected {expected_calls} process_board_data calls, got {actual_calls}")
        
        print("✓ Stream manager was used correctly")
        print(f"✓ Data was processed {actual_calls} times")
        print("✓ Pipeline completed successfully")
        
        print("\nStream restart test passed! ✨")

    def test_mock_setup(self):
        """Test that all mocks are properly configured."""
        print("\nTesting mock setup...")
        
        # Test BoardManager mock
        self.assertEqual(self.mock_board_manager.sampling_rate, 125)
        self.assertEqual(self.mock_board_manager.board_timestamp_channel, 30)
        self.assertEqual(self.mock_board_manager.board_shim.get_board_id(), BoardIds.CYTON_DAISY_BOARD)
        self.assertEqual(self.mock_board_manager.board_shim.get_exg_channels(), list(range(8)))
        print("✓ BoardManager mock configured correctly")
        
        # Test DataManager mock
        self.assertTrue(self.mock_data_manager.queue_data_for_csv_write.return_value)
        self.assertTrue(self.mock_data_manager.add_to_data_processing_buffer.return_value)
        self.mock_data_manager.save_new_data.assert_not_called()
        self.assertEqual(self.mock_data_manager._calculate_next_buffer_id_to_process(), 0)
        can_process, reason, start_idx, end_idx = self.mock_data_manager.next_available_epoch_on_round_robin_buffer(0)
        self.assertTrue(can_process)
        self.assertEqual(end_idx - start_idx, 1250)  # 10 seconds at 125 Hz
        print("✓ DataManager mock configured correctly")
        
        # Test ReceivedStreamedDataHandler mock
        self.mock_data_handler.process_board_data.assert_not_called()
        self.assertEqual(self.mock_data_handler.data_manager, self.mock_data_manager)
        print("✓ ReceivedStreamedDataHandler mock configured correctly")
        
        # Test logger mock
        self.mock_logger.info.assert_not_called()
        self.mock_logger.error.assert_not_called()
        self.mock_logger.warning.assert_not_called()
        print("✓ Logger mock configured correctly")
        
        # Test Qt App mock
        self.mock_qt_app.processEvents.assert_not_called()
        print("✓ Qt App mock configured correctly")
        
        # Test Visualizer mock
        self.mock_visualizer.app.assert_not_called()
        print("✓ Visualizer mock configured correctly")
        
        print("\nAll mock tests passed! ✨")

if __name__ == '__main__':
    print("\nRunning tests directly...")
    unittest.main(verbosity=2) 