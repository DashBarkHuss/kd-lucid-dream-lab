"""
Script to copy and cut a portion of real data into a smaller file.
This script takes a real data file and creates a smaller version by cutting a specific time window.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def copy_and_cut_data(input_file, output_file, start_time=30, duration=60):
    """
    Copy and cut a portion of real data into a smaller file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        start_time (float): Start time in seconds
        duration (float): Duration in seconds to cut
    """
    # Read the input file
    data = pd.read_csv(input_file, sep='\t', header=None)
    
    # Calculate sample indices
    sampling_rate = 125  # BrainFlow default for Cyton+Daisy
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + duration) * sampling_rate)
    
    # Cut the data
    test_data = data.iloc[start_sample:end_sample]
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the test data
    test_data.to_csv(output_file, sep='\t', header=False, index=False)
    
    print(f"Created test data file at: {output_file}")
    print(f"Duration: {duration} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {len(test_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and cut a portion of real data into a smaller file')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--start', type=float, default=30, help='Start time in seconds (default: 30)')
    parser.add_argument('--duration', type=float, default=60, help='Duration in seconds to cut (default: 60)')
    
    args = parser.parse_args()
    
    copy_and_cut_data(args.input_file, args.output_file, args.start, args.duration) 