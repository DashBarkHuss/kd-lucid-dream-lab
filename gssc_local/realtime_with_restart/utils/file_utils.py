#!/usr/bin/env python3

"""
File manipulation utilities for the realtime_with_restart module.

This module provides common file operations used across different
main files and components in the system.
"""


def create_trimmed_csv(input_file: str, output_file: str, skip_samples: int) -> None:
    """Create a new CSV file starting from the specified sample offset.
    
    This utility function is commonly used in gap handling scenarios where
    we need to restart data processing from a specific point in the original file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file  
        skip_samples (int): Number of lines to skip from the beginning
        
    Examples:
        >>> create_trimmed_csv('data.csv', 'trimmed_data.csv', 1000)
        # Creates trimmed_data.csv starting from line 1001 of data.csv
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            if idx >= skip_samples:
                outfile.write(line)