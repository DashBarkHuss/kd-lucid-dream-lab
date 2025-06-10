import numpy as np
import pandas as pd
import time
from pathlib import Path
from brainflow.board_shim import BoardShim, BoardIds
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gssc_local.realtime_with_restart.export.csv.utils import create_format_string

def create_consecutive_data(output_file, duration=30):
    """
    Create a BrainFlow-compatible CSV file with consecutive numbers in all EEG channels
    
    Args:
        output_file: Path to save the output CSV file
        duration: Total duration of data in seconds
    """
    # Get board configuration
    master_board_id = BoardIds.CYTON_DAISY_BOARD
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)  # 125 Hz
    timestamp_channel = BoardShim.get_timestamp_channel(master_board_id)
    
    # Calculate number of samples
    n_samples = int(duration * sampling_rate)
    
    # Initialize data array with 32 columns (BrainFlow format)
    data = np.zeros((n_samples, 32))
    
    # Column 1: Package number (increments by 1)
    data[:, 0] = np.arange(n_samples) + 2  # Start at 2 like the example
    
    # Column 2: Sample index (increments by 1)
    data[:, 1] = np.arange(1, n_samples + 1)
    
    # Columns 3-17: Consecutive numbers starting from 2916
    base_value = 2916  # Starting value from example file
    for i in range(2, 17):
        data[:, i] = base_value + np.arange(n_samples)
    
    # Columns 18-20: EEG/signal data (consecutive numbers)
    for i in range(17, 20):
        data[:, i] = np.arange(n_samples)
    
    # Column 21: Fixed value
    data[:, 20] = 192.0
    
    # Columns 22-29: Zeros (already initialized as zeros)
    
    # Set timestamps (no gaps)
    start_time = time.time()
    expected_interval = 1.0 / sampling_rate
    timestamps = start_time + np.arange(n_samples) * expected_interval
    data[:, timestamp_channel] = timestamps
    
    # Ensure the directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Create format string for each column (32 columns total)
    fmt = create_format_string(32)
    
    # Save to CSV in BrainFlow format
    np.savetxt(
        str(output_file),
        data,
        delimiter='\t',
        fmt=fmt
    )
    
    print(f"Created data file at: {output_file}")
    print(f"Total duration: {duration} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {n_samples}")
    print(f"Starting value: {base_value}")
    
    return output_file

if __name__ == "__main__":
    # Create test data with 30 seconds of consecutive numbers
    output_file = "data/test_data/consecutive_data.csv"
    create_consecutive_data(output_file, duration=35)