import time
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from .board_manager import BoardManager

class MockBoardManager(BoardManager):
    """Mock implementation that only overrides the streaming functionality."""
    
    def __init__(self, csv_file_path: str, speed_multiplier: float = 10.0):
        # Initialize the real board manager first
        super().__init__(file_path=csv_file_path)
        
        # Add our mock-specific attributes
        self.speed_multiplier = speed_multiplier
        self.file_data = None
        self.current_position = 0
        self.start_time = None
        self.last_chunk_time = None

    def setup_board(self):
        # Load test data using numpy for more reliable reading
        self.file_data = pd.read_csv(self.file_path, sep='\t', header=None, dtype=float)
        print(f"[DEBUG] setup_board: After loading, shape: {self.file_data.shape}")
        print(f"[DEBUG] Last 3 rows:\n{self.file_data.tail(3)}")
        
        # Setup real board (but we won't use its streaming)
        print(f"[DEBUG] setup_board: Before super().setup_board(), shape: {self.file_data.shape}")
        result = super().setup_board()
        print(f"[DEBUG] setup_board: After super().setup_board(), shape: {self.file_data.shape}")
        return result

    def start_stream(self):
        # Start our mock streaming
        self.current_position = 0
        self.start_time = time.time()
        self.last_chunk_time = self.start_time
        return len(self.file_data)

    def get_initial_data(self):
        print(f"[DEBUG] get_initial_data: self.file_data.shape before chunking: {self.file_data.shape}")
        # Return a full chunk if possible, or as many as are available
        chunk_size = self.sampling_rate  # One second worth of data
        points_to_return = min(chunk_size, len(self.file_data))
        print(f"[DEBUG] get_initial_data: points_to_return={points_to_return}")
        if points_to_return > 0:
            data = self.file_data.iloc[:points_to_return].values.T
            print(f"[DEBUG] get_initial_data: returning samples 0:{points_to_return} (shape={data.shape})")
            self.current_position = points_to_return
            return data
        return np.array([])

    def get_new_data(self):
        """Get new data from the board"""
        chunk_size = self.sampling_rate  # One second worth of data
        
        # If we're at the end, return empty array
        if self.current_position >= len(self.file_data):
            return np.array([])
            
        # Calculate how many samples to return
        remaining_samples = len(self.file_data) - self.current_position
        samples_to_return = min(chunk_size, remaining_samples)
        
        # Get the data for this chunk
        data = self.file_data.iloc[self.current_position:self.current_position + samples_to_return].values.T
        
        # Update position
        self.current_position += samples_to_return
        
        # Only sleep if we're not at the end and we got a full chunk
        if self.current_position < len(self.file_data) and samples_to_return == chunk_size:
            # Sleep for 1/speed_multiplier seconds between chunks
            time.sleep(1.0 / self.speed_multiplier)
            
        return data







