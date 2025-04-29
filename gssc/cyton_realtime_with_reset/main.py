#!/usr/bin/env python3

import sys
import os
import logging
import argparse

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Setting up logging...")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

print("Importing manager...")
from cyton_realtime_with_reset.app.manager import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Cyton realtime processing with gap detection')
    parser.add_argument('--csv-file', type=str, help='Path to the CSV file to process', 
                       default="data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv")
    args = parser.parse_args()
    
    print(f"Starting application with file: {args.csv_file}")
    main(args.csv_file)