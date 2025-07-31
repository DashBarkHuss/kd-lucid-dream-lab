import time
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from .board_manager import BoardManager

class SpeedControlledBoardManager(BoardManager):
    """Speed-controlled implementation of BoardManager that allows configurable playback speed.
    
    This class inherits from BoardManager and overrides the streaming functionality to provide
    a controlled playback of data from a CSV file. It's useful for testing and development
    without requiring the testing with true to life timing, which can be useful for testing
    data files that are long in duration.
    
    Key differences from real board:
    1. Configurable playback speed through speed_multiplier
    2. Does not wait for gaps in data - plays through them at the specified speed
       (unlike real board which would wait for the actual gap duration)
    
    Attributes:
        speed_multiplier (float): Factor by which to speed up the data playback (default: 10.0)
        file_data (pd.DataFrame): Loaded CSV data for playback
        current_position (int): Current position in the data stream
        start_time (float): Timestamp when streaming started
        last_chunk_time (float): Timestamp of the last data chunk
    """
    
    def __init__(self, csv_file_path: str, speed_multiplier: float = 10.0):
        """Initialize the SpeedControlledBoardManager.
        
        Args:
            csv_file_path (str): Path to the CSV file containing the data to stream
            speed_multiplier (float, optional): Factor to speed up playback. Defaults to 10.0.
                A value of 1.0 means real-time speed, 2.0 means twice as fast, etc.
        """
        # Initialize the real board manager first to get base functionality
        super().__init__()
        
        # Add our mock-specific attributes
        self.speed_multiplier = speed_multiplier
        self.file_path = csv_file_path
        self.file_data = None
        self.current_position = 0
        self.start_time = None
        self.last_chunk_time = None
        
        # Gap simulation attributes
        self.in_gap_mode = False
        self.gap_start_real_time = None
        self.gap_duration_seconds = None
        self.expected_timestamp = None

    def set_board_shim(self):
        """Initialize the mock board for data collection.
        
        This method:
        1. Loads the CSV file data using pandas with tab separator and float data types
        2. Calls the parent BoardManager's set_board_shim() method to handle actual board configuration
        
        Returns:
            BoardShim: The configured board shim object
        
        Raises:
            Exception: If file loading or board setup fails
        """
        # Load test data using pandas for reliable CSV reading
        # Using tab separator and no header to match real board format
        # dtype=float ensures consistent data type handling
        self.file_data = pd.read_csv(self.file_path, sep='\t', header=None, dtype=float)
        print(f"[DEBUG] set_board_shim: After loading, shape: {self.file_data.shape}")
        print(f"[DEBUG] Last 3 rows:\n{self.file_data.tail(3)}")
        
        # Setup real board (but we won't use its streaming)
        # This ensures we have all the necessary board configuration
        print(f"[DEBUG] set_board_shim: Before super().set_board_shim(), shape: {self.file_data.shape}")
        result = super().set_board_shim()
        print(f"[DEBUG] set_board_shim: After super().set_board_shim(), shape: {self.file_data.shape}")
        return result

    def start_stream(self):
        """Start the mock data stream.
        
        This method initializes the streaming state and prepares for data playback.
        It resets the current position and sets up timing variables based on the
        first timestamp in the file data.
        
        Returns:
            int: Total number of samples available in the data file
            
        Raises:
            ValueError: If no timestamps are available in the file data
        """
        # Reset position to start of file
        self.current_position = 0
        
        # Initialize timing variables using the first timestamp from the file
        if self.board_timestamp_channel is None or len(self.file_data) == 0:
            raise ValueError("No timestamps available in file data")
            
        self.start_time = float(self.file_data.iloc[0, self.board_timestamp_channel])
        self.last_chunk_time = self.start_time
        # Initialize expected timestamp to None - will be set after first chunk
        self.expected_timestamp = None
        
        # Return total available samples for progress tracking
        return len(self.file_data)

    def get_initial_data(self):
        """Get the initial chunk of data from the stream.
        
        This method returns the first chunk of data (one second worth) from the
        loaded CSV file. It's used to initialize the data stream.
        
        Returns:
            numpy.ndarray: Initial data chunk with shape (channels, samples)
        """
        print(f"[DEBUG] get_initial_data: self.file_data.shape before chunking: {self.file_data.shape}")
        
        # Calculate chunk size based on sampling rate (1 second of data)
        chunk_size = self.sampling_rate
        
        # Ensure we don't try to read past the end of the file
        points_to_return = min(chunk_size, len(self.file_data))
        print(f"[DEBUG] get_initial_data: points_to_return={points_to_return}")
        
        if points_to_return > 0:
            # Get first chunk of data and transpose to match real board format
            # Real board returns data in shape (channels, samples)
            data = self.file_data.iloc[:points_to_return].values.T
            print(f"[DEBUG] get_initial_data: returning samples 0:{points_to_return} (shape={data.shape})")
            
            # Update position for next read
            self.current_position = points_to_return
            return data
            
        # Return empty array if no data available
        return np.array([])

    def get_new_data(self):
        """Get the next chunk of data from the stream.
        
        This method returns the next chunk of data based on the current position
        and sampling rate. It implements the speed multiplier by controlling the
        timing between chunks and simulates gaps by detecting timestamp jumps.
        
        Returns:
            numpy.ndarray: Next data chunk with shape (channels, samples), or empty array during gaps
        """
        # Calculate chunk size based on sampling rate (1 second of data)
        chunk_size = self.sampling_rate
        
        # Check if we're currently in gap simulation mode
        if self.in_gap_mode:
            # Check if the gap duration has elapsed (accounting for speed multiplier)
            elapsed_real_time = time.time() - self.gap_start_real_time
            gap_duration_real_time = self.gap_duration_seconds / self.speed_multiplier
            
            if elapsed_real_time >= gap_duration_real_time:
                # Gap is over, exit gap mode
                print(f"[DEBUG] Gap simulation complete. Elapsed: {elapsed_real_time:.3f}s, Expected: {gap_duration_real_time:.3f}s")
                self.in_gap_mode = False
                self.gap_start_real_time = None
                self.gap_duration_seconds = None
                # Reset expected timestamp so we don't re-detect the same gap
                self.expected_timestamp = None
                # Continue with normal data processing below
            else:
                # Still in gap, return empty array
                print(f"[DEBUG] Still in gap mode. Elapsed: {elapsed_real_time:.3f}s / {gap_duration_real_time:.3f}s")
                return np.array([])
        
        # Check if we've reached the end of the file
        if self.current_position >= len(self.file_data):
            return np.array([])
            
        # Calculate how many samples to return
        remaining_samples = len(self.file_data) - self.current_position
        samples_to_return = min(chunk_size, remaining_samples)
        
        # Get the current timestamp to check for gaps
        current_timestamp = float(self.file_data.iloc[self.current_position, self.board_timestamp_channel])
        
        # Reason: Gap detection - check if there's a significant jump in timestamps
        if self.expected_timestamp is not None:
            expected_sample_duration = samples_to_return / self.sampling_rate  # Duration of this chunk
            timestamp_diff = current_timestamp - self.expected_timestamp
            
            # If timestamp jump is more than 1.5x the expected interval between samples, consider it a gap
            expected_interval = 1.0 / self.sampling_rate  # Time between individual samples
            if timestamp_diff > (expected_interval * 1.5):
                # Gap detected! Enter gap simulation mode
                print(f"[DEBUG] Gap detected! Expected: {self.expected_timestamp}, Got: {current_timestamp}, Diff: {timestamp_diff}")
                self.in_gap_mode = True
                self.gap_start_real_time = time.time()
                self.gap_duration_seconds = timestamp_diff
                
                # Don't update expected_timestamp yet - let the gap mode handle it
                # Return empty array to simulate gap
                return np.array([])
        
        # Get the data for this chunk and transpose to match real board format
        data = self.file_data.iloc[self.current_position:self.current_position + samples_to_return].values.T
        
        # Update position for next read
        self.current_position += samples_to_return
        
        # Update expected timestamp for next chunk
        if samples_to_return > 0:
            sample_duration = samples_to_return / self.sampling_rate
            self.expected_timestamp = current_timestamp + sample_duration
        
        # Sleep for timing control (normal playback speed)
        if self.current_position < len(self.file_data) and samples_to_return == chunk_size:
            time.sleep(1.0 / self.speed_multiplier)
            
        return data







