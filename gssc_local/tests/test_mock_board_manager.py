import unittest
import pandas as pd
import numpy as np
import sys
import os
import time
from brainflow.board_shim import BoardShim, BoardIds
import pytest

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from gssc_local.realtime_with_restart.mock_board_manager import MockBoardManager
from gssc_local.tests.test_utils import create_brainflow_test_data, save_brainflow_data_to_csv
from gssc_local.realtime_with_restart.data_manager import DataManager

class TestMockBoardManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Ensure all board sessions are released before starting a new test
        BoardShim.release_all_sessions()
        # Create 20 seconds of test data to handle slower speed multipliers
        # For speed_multiplier=0.5, we need at least 10 seconds to get 3 full chunks
        data, metadata = create_brainflow_test_data(duration_seconds=20.0)
        
        # Save to CSV in BrainFlow format
        self.csv_path = 'test_data.csv'
        save_brainflow_data_to_csv(data, self.csv_path)
        # Print the number of lines in the CSV file for debugging
        with open(self.csv_path, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"[DEBUG] test_data.csv line count after save: {line_count}")
        
        # Initialize the mock with test data
        self.mock = MockBoardManager(csv_file_path=self.csv_path, speed_multiplier=1.0)
        
        # Store metadata for assertions
        self.sampling_rate = metadata['sampling_rate']
        self.timestamp_channel = metadata['timestamp_channel']
        self.start_time = metadata['start_time']
        self.n_samples = metadata['n_samples']

    def test_initial_data(self):
        """Test that initial data is returned correctly."""
        # Setup the board first
        self.mock.setup_board()
        
        # Start the stream
        self.mock.start_stream()
        
        # Get initial data
        initial_data = self.mock.get_initial_data()
        
        # Verify data shape and content
        self.assertIsNotNone(initial_data)
        self.assertIsInstance(initial_data, np.ndarray)
        
        # Data should be in channels x samples format
        self.assertEqual(initial_data.shape[0], len(self.mock.file_data.columns))
        
        # First sample should match the first row of our test data
        expected_first_sample = self.mock.file_data.iloc[0].values
        np.testing.assert_array_almost_equal(initial_data[:, 0], expected_first_sample)
        
        # Verify timestamp channel
        timestamp_channel = self.mock.timestamp_channel
        self.assertIsNotNone(timestamp_channel)
        self.assertIn(timestamp_channel, range(initial_data.shape[0]))
        
        # Verify timestamps are increasing
        timestamps = initial_data[timestamp_channel]
        self.assertTrue(np.all(np.diff(timestamps) > 0))

    def test_streaming_speed(self):
        """Test that data is streamed at the correct speed."""
        try:
            # Setup the board first
            self.mock.setup_board()
            
            # Start the stream
            self.mock.start_stream()
            
            # Get initial data
            initial_data = self.mock.get_initial_data()
            
            # Record start time
            start_time = time.time()
            
            # Get multiple chunks of data and measure time between them
            chunks = []
            times = []
            
            for _ in range(3):  # Get 3 chunks to verify consistent timing
                chunk_start = time.time()
                data = self.mock.get_new_data()
                chunk_end = time.time()
                
                if data.size > 0:
                    chunks.append(data)
                    times.append(chunk_end - chunk_start)
                
                # Small sleep to ensure we get distinct chunks
                time.sleep(0.1)
            
            # Calculate average time between chunks
            if len(times) > 1:
                avg_time = sum(times) / len(times)
                
                # Calculate expected time based on speed multiplier
                # For speed_multiplier=1.0, we expect roughly 1 second between chunks
                # since the mock sleeps for 1 second between chunks
                expected_time = 1.0 / self.mock.speed_multiplier
                
                # Allow for some timing variance (20% tolerance)
                tolerance = 0.2
                self.assertAlmostEqual(avg_time, expected_time, delta=expected_time * tolerance,
                                     msg=f"Average time between chunks ({avg_time:.3f}s) does not match "
                                         f"expected time ({expected_time:.3f}s) with speed multiplier "
                                         f"{self.mock.speed_multiplier}")
                
                # Verify we got data in each chunk
                self.assertTrue(all(chunk.size > 0 for chunk in chunks),
                              "Some chunks contained no data")
                
                # Verify data shape is consistent
                if len(chunks) > 1:
                    first_shape = chunks[0].shape
                    self.assertTrue(all(chunk.shape == first_shape for chunk in chunks),
                                  "Chunk shapes are inconsistent")
                    
                    # Verify chunk sizes are reasonable
                    # For speed_multiplier=1.0, we expect roughly 125 samples per second
                    expected_samples = int(self.sampling_rate / self.mock.speed_multiplier)
                    tolerance_samples = int(expected_samples * 0.2)  # 20% tolerance
                    for chunk in chunks:
                        self.assertAlmostEqual(chunk.shape[1], expected_samples, 
                                             delta=tolerance_samples,
                                             msg=f"Chunk size {chunk.shape[1]} does not match expected size {expected_samples}")
            else:
                self.fail("Not enough data chunks were collected to verify timing")
        finally:
            # Ensure board is cleaned up even if test fails
            if hasattr(self, 'mock') and self.mock.board_shim is not None:
                try:
                    self.mock.board_shim.release_session()
                except Exception as e:
                    print(f"Warning: Error releasing board session: {e}")

    def test_data_format(self):
        """Test that returned data matches the format expected by the main pipeline."""
        # Setup the board first
        self.mock.setup_board()
        
        # Start the stream
        self.mock.start_stream()
        
        # Get initial data
        initial_data = self.mock.get_initial_data()
        
        # Verify data format
        self.assertIsNotNone(initial_data)
        self.assertIsInstance(initial_data, np.ndarray)
        
        # Data should be in channels x samples format
        self.assertEqual(initial_data.shape[0], len(self.mock.file_data.columns))
        
        # Get a chunk of new data
        new_data = self.mock.get_new_data()
        
        # Verify new data format
        self.assertIsNotNone(new_data)
        self.assertIsInstance(new_data, np.ndarray)
        
        # Data should be in channels x samples format
        self.assertEqual(new_data.shape[0], len(self.mock.file_data.columns))
        
        # Verify timestamp channel exists and contains increasing values
        timestamp_channel = self.mock.timestamp_channel
        self.assertIsNotNone(timestamp_channel)
        self.assertIn(timestamp_channel, range(initial_data.shape[0]))
        
        # Verify timestamps are increasing
        timestamps = initial_data[timestamp_channel]
        self.assertTrue(np.all(np.diff(timestamps) > 0))
        
        # Verify data types are float
        self.assertTrue(np.issubdtype(initial_data.dtype, np.floating))
        self.assertTrue(np.issubdtype(new_data.dtype, np.floating))
        
        # Verify no NaN values
        self.assertFalse(np.any(np.isnan(initial_data)))
        self.assertFalse(np.any(np.isnan(new_data)))
        
        # Get EEG channels from board configuration
        eeg_channels = BoardShim.get_eeg_channels(self.mock.board_shim.get_board_id())
        
        # Verify EEG data values are within reasonable range
        # For test data, we expect sine waves between -1 and 1
        for channel in eeg_channels:
            self.assertTrue(np.all(np.abs(initial_data[channel]) <= 1.0),
                          f"EEG channel {channel} contains values outside [-1, 1] range")
            self.assertTrue(np.all(np.abs(new_data[channel]) <= 1.0),
                          f"EEG channel {channel} contains values outside [-1, 1] range")

    def test_speed_multiplier(self):
        """Test that different speed multipliers work correctly."""
        # Test different speed multipliers
        speed_multipliers = [10.0, 100.0]
        
        # Store processing times for each speed
        processing_times = {}
        
        for speed_mult in speed_multipliers:
            print(f"\nTesting speed multiplier: {speed_mult}")
            # Create new mock with current speed multiplier
            mock = MockBoardManager(csv_file_path=self.csv_path, speed_multiplier=speed_mult)
            
            try:
                # Setup and start stream
                mock.setup_board()
                mock.start_stream()
                
                # Get initial data
                initial_data = mock.get_initial_data()
                print(f"Initial data shape: {initial_data.shape if initial_data is not None else 'None'}")
                
                # Record start time
                start_time = time.time()
                total_samples = initial_data.shape[1] if initial_data is not None else 0
                print(f"Starting with {total_samples} samples from initial data")
                
                # Process entire file with timeout
                max_iterations = len(mock.file_data) * 2  # Safety limit
                iteration = 0
                
                while iteration < max_iterations:
                    data = mock.get_new_data()
                    if data.size == 0:
                        print(f"Empty data returned at iteration {iteration}")
                        break
                    total_samples += data.shape[1]
                    print(f"Iteration {iteration}: Added {data.shape[1]} samples, total now {total_samples}")
                    iteration += 1
                
                if iteration >= max_iterations:
                    self.fail(f"Test timed out after {max_iterations} iterations")
                
                # Calculate total processing time
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times[speed_mult] = processing_time
                
                print(f"\nSpeed multiplier {speed_mult}:")
                print(f"Total samples processed: {total_samples}")
                print(f"Total processing time: {processing_time:.2f} seconds")
                print(f"Effective samples per second: {total_samples/processing_time:.2f}")
                
                # Verify we processed all samples
                self.assertEqual(total_samples, len(mock.file_data),
                               f"Speed multiplier {speed_mult}: Did not process all samples")
                
            finally:
                # Clean up
                if mock.board_shim is not None:
                    try:
                        mock.board_shim.release_session()
                    except Exception as e:
                        print(f"Warning: Error releasing board session: {e}")
        
        # Use the lowest speed multiplier as the baseline
        baseline_speed = speed_multipliers[0]
        reference_time = processing_times[baseline_speed]
        for speed_mult in speed_multipliers[1:]:  # Compare higher speeds to baseline
            expected_time = reference_time * (baseline_speed / speed_mult)
            actual_time = processing_times[speed_mult]
            # Allow for 50% tolerance
            tolerance = 0.5
            self.assertLess(actual_time, expected_time * (1 + tolerance),
                          f"Speed multiplier {speed_mult}: Processing time {actual_time:.2f}s "
                          f"exceeds expected time {expected_time:.2f}s by more than {tolerance*100}%")
            self.assertGreater(actual_time, expected_time * (1 - tolerance),
                             f"Speed multiplier {speed_mult}: Processing time {actual_time:.2f}s "
                             f"is less than expected time {expected_time:.2f}s by more than {tolerance*100}%")

    def test_end_of_data(self):
        """Test behavior when reaching the end of the data file."""
        # Ensure all board sessions are released before starting this test
        BoardShim.release_all_sessions()
        try:
            # Setup the board first
            self.mock.setup_board()
            
            # Start the stream
            self.mock.start_stream()
            
            # Get initial data
            initial_data = self.mock.get_initial_data()
            
            # Calculate how many chunks we need to read to reach the end
            chunk_size = self.sampling_rate  # One second worth of data
            total_samples = len(self.mock.file_data)
            initial_samples = initial_data.shape[1] if initial_data.size > 0 else 0
            remaining_samples = total_samples - initial_samples
            expected_chunks = (remaining_samples + chunk_size - 1) // chunk_size  # Ceiling division
            
            # Read chunks until we reach the end
            chunks = []
            for _ in range(expected_chunks + 1):  # Read one extra chunk to verify end behavior
                data = self.mock.get_new_data()
                if data.size > 0:
                    chunks.append(data)
            
            # Verify we got the expected number of chunks
            self.assertEqual(len(chunks), expected_chunks, 
                            f"Expected {expected_chunks} chunks but got {len(chunks)}")
            
            # Verify the last chunk contains the remaining samples
            last_chunk = chunks[-1]
            expected_last_chunk_size = remaining_samples % chunk_size
            if expected_last_chunk_size == 0:
                expected_last_chunk_size = chunk_size
            self.assertEqual(last_chunk.shape[1], expected_last_chunk_size,
                            f"Last chunk size {last_chunk.shape[1]} does not match expected size {expected_last_chunk_size}")
            
            # Verify that reading beyond the end returns empty array
            empty_data = self.mock.get_new_data()
            self.assertEqual(empty_data.size, 0, 
                            "Expected empty array when reading beyond end of data")
            
            # Verify that the current position is at the end
            self.assertEqual(self.mock.current_position, total_samples,
                            f"Current position {self.mock.current_position} should be at end {total_samples}")
            
            # Verify that timestamps in the last chunk are correct
            if self.timestamp_channel is not None:
                last_chunk_timestamps = last_chunk[self.timestamp_channel]
                expected_last_timestamp = self.mock.file_data.iloc[-1].iloc[self.timestamp_channel]
                self.assertAlmostEqual(last_chunk_timestamps[-1], expected_last_timestamp,
                                     msg="Last timestamp does not match expected value")
        finally:
            # Ensure board is cleaned up even if test fails
            if hasattr(self, 'mock') and self.mock.board_shim is not None:
                try:
                    self.mock.board_shim.release_session()
                except Exception as e:
                    print(f"Warning: Error releasing board session: {e}")

    def test_csv_output(self):
        """Test that all data from the mock is correctly saved to CSV."""
        # Setup the board first
        self.mock.setup_board()
        
        # Create DataManager
        data_manager = DataManager(self.mock.board_shim, self.sampling_rate)
        
        # Start the stream
        self.mock.start_stream()
        
        # Get initial data
        initial_data = self.mock.get_initial_data()
        print(f"\n[DEBUG] Initial data shape: {initial_data.shape if initial_data is not None else 'None'}")
        print(f"[DEBUG] Current position after initial data: {self.mock.current_position}")
        
        if initial_data.size > 0:
            data_manager.add_data(initial_data, is_initial=True)
            data_manager.save_new_data(initial_data, is_initial=True)
            print(f"[DEBUG] Saved initial data with shape: {initial_data.shape}")
        
        # Process all data
        total_samples = initial_data.shape[1] if initial_data.size > 0 else 0
        print(f"[DEBUG] Starting with {total_samples} samples from initial data")
        
        while True:
            data = self.mock.get_new_data()
            if data.size == 0:
                break
            print(f"[DEBUG] Got new data chunk with shape: {data.shape}")
            data_manager.add_data(data)
            data_manager.save_new_data(data)
            total_samples += data.shape[1]
            print(f"[DEBUG] Total samples processed: {total_samples}")
        
        print(f"[DEBUG] Final total samples: {total_samples}")
        print(f"[DEBUG] Expected total samples: {len(self.mock.file_data)}")
        
        # Save final data to CSV
        output_file = 'test_output.csv'
        data_manager.save_to_csv(output_file)
        
        try:
            # Verify the data matches exactly
            original_data = pd.read_csv(self.csv_path, sep='\t', header=None)
            saved_data = pd.read_csv(output_file, sep='\t', header=None)
            
            print(f"\n[DEBUG] Original data shape: {original_data.shape}")
            print(f"[DEBUG] Saved data shape: {saved_data.shape}")
            
            # Compare line counts
            if len(original_data) != len(saved_data):
                print(f"Line count mismatch: Original={len(original_data)}, Saved={len(saved_data)}")
                # Find which index is missing
                min_len = min(len(original_data), len(saved_data))
                for i in range(min_len):
                    if not original_data.iloc[i].equals(saved_data.iloc[i]):
                        print(f"First mismatch at row {i}:")
                        print(f"Original: {original_data.iloc[i].values}")
                        print(f"Saved:    {saved_data.iloc[i].values}")
                        break
                else:
                    if len(original_data) > len(saved_data):
                        print(f"Missing row at index {min_len}: {original_data.iloc[min_len].values}")
                    else:
                        print(f"Extra row at index {min_len}: {saved_data.iloc[min_len].values}")
            self.assertEqual(len(original_data), len(saved_data),
                           f"Line count mismatch: Original={len(original_data)}, Saved={len(saved_data)}")
            
            # Compare actual data
            try:
                pd.testing.assert_frame_equal(original_data, saved_data.iloc[:, :-2], check_dtype=False)
            except AssertionError as e:
                print("Data mismatch detected:")
                print(e)
                # Print the first row where data differs
                for i in range(len(saved_data)):
                    if not original_data.iloc[i].equals(saved_data.iloc[i, :-2]):
                        print(f"First data mismatch at row {i}:")
                        print(f"Original: {original_data.iloc[i].values}")
                        print(f"Saved:    {saved_data.iloc[i, :-2].values}")
                        break
                raise
            
        finally:
            # Clean up
            if os.path.exists(output_file):
                os.remove(output_file)
            data_manager.cleanup()

    def test_csv_row_integrity(self):
        """Test that the number of rows in the CSV matches the number loaded with np.loadtxt."""
        # Count lines in the CSV file
        with open(self.csv_path, 'r') as f:
            csv_line_count = sum(1 for _ in f)
        print(f"[DEBUG] test_csv_row_integrity: CSV file line count: {csv_line_count}")

        # Load with np.loadtxt
        loaded_data = np.loadtxt(self.csv_path, delimiter='\t', dtype=float)
        print(f"[DEBUG] test_csv_row_integrity: np.loadtxt loaded shape: {loaded_data.shape}")

        # If 1D, reshape to (1, N)
        if loaded_data.ndim == 1:
            loaded_data = loaded_data.reshape(1, -1)

        self.assertEqual(csv_line_count, loaded_data.shape[0],
            f"CSV file has {csv_line_count} lines, but np.loadtxt loaded {loaded_data.shape[0]} rows")

    def test_data_shape_through_flow(self):
        """Test that the data shape remains consistent through the entire flow."""
        # First verify CSV and initial load
        with open(self.csv_path, 'r') as f:
            csv_line_count = sum(1 for _ in f)
        print(f"[DEBUG] test_data_shape_through_flow: CSV file line count: {csv_line_count}")
        
        # Load with np.loadtxt
        loaded_data = np.loadtxt(self.csv_path, delimiter='\t', dtype=float)
        print(f"[DEBUG] test_data_shape_through_flow: np.loadtxt loaded shape: {loaded_data.shape}")
        
        # Initialize mock and check shape after setup_board
        mock = MockBoardManager(csv_file_path=self.csv_path, speed_multiplier=1.0)
        mock.setup_board()
        print(f"[DEBUG] test_data_shape_through_flow: shape after setup_board: {mock.file_data.shape}")
        
        # Start stream and check shape
        mock.start_stream()
        print(f"[DEBUG] test_data_shape_through_flow: shape after start_stream: {mock.file_data.shape}")
        
        # Get initial data and check shape
        initial_data = mock.get_initial_data()
        print(f"[DEBUG] test_data_shape_through_flow: initial_data shape: {initial_data.shape if initial_data is not None else 'None'}")
        print(f"[DEBUG] test_data_shape_through_flow: current_position after initial data: {mock.current_position}")
        
        # Assertions
        self.assertEqual(csv_line_count, loaded_data.shape[0], "CSV line count doesn't match loaded data")
        self.assertEqual(loaded_data.shape[0], mock.file_data.shape[0], "Data shape changed after setup_board")
        self.assertEqual(mock.file_data.shape[0], 2500, "Data shape should be 2500 rows")

    def test_csv_loading_methods(self):
        """Test that both pd.read_csv and np.loadtxt load the same number of rows."""
        # Count lines in the CSV file
        with open(self.csv_path, 'r') as f:
            csv_line_count = sum(1 for _ in f)
        print(f"[DEBUG] test_csv_loading_methods: CSV file line count: {csv_line_count}")
        
        # Load with np.loadtxt
        np_data = np.loadtxt(self.csv_path, delimiter='\t', dtype=float)
        print(f"[DEBUG] test_csv_loading_methods: np.loadtxt loaded shape: {np_data.shape}")
        
        # Load with pd.read_csv
        pd_data = pd.read_csv(self.csv_path, sep='\t', dtype=float, header=None)
        print(f"[DEBUG] test_csv_loading_methods: pd.read_csv loaded shape: {pd_data.shape}")
        
        # Print first and last few rows of each
        print("\nFirst 3 rows from np.loadtxt:")
        print(np_data[:3])
        print("\nFirst 3 rows from pd.read_csv:")
        print(pd_data.head(3))
        
        print("\nLast 3 rows from np.loadtxt:")
        print(np_data[-3:])
        print("\nLast 3 rows from pd.read_csv:")
        print(pd_data.tail(3))
        
        # Assertions
        self.assertEqual(csv_line_count, np_data.shape[0], 
            f"np.loadtxt loaded {np_data.shape[0]} rows, expected {csv_line_count}")
        self.assertEqual(csv_line_count, len(pd_data), 
            f"pd.read_csv loaded {len(pd_data)} rows, expected {csv_line_count}")
        self.assertEqual(np_data.shape[0], len(pd_data),
            f"np.loadtxt loaded {np_data.shape[0]} rows, pd.read_csv loaded {len(pd_data)} rows")

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test files
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
            
        # Clean up board if it exists
        if hasattr(self, 'mock') and self.mock.board_shim is not None:
            try:
                self.mock.board_shim.release_session()
            except Exception as e:
                print(f"Warning: Error releasing board session: {e}")

if __name__ == '__main__':
    print("\nRunning tests directly...")
    # run with logs
    pytest.main([__file__, '-v', '-s']) 