"""Utility functions for CSV operations."""

import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple
import os
import numpy as np
import pandas as pd
from .exceptions import CSVDataError, CSVExportError, CSVFormatError
from .validation import (
    validate_csv_not_empty, 
    validate_transformed_rows_not_empty,
    find_duplicates,
    validate_sleep_stage_timestamps
)

# Format specifier for main BrainFlow data:
# - all columns use 6 decimal places for consistency
MAIN_DATA_FMT = '%.6f'

def check_file_exists(file_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> bool:
    """Check if a file exists.
    
    Args:
        file_path (Union[str, Path]): Path to the file to check
        logger (Optional[logging.Logger]): Logger for info messages
        
    Returns:
        bool: True if file exists, False otherwise
    """
    path = Path(file_path)
    exists = os.path.exists(path)
    if logger and not exists:
        logger.info(f"File not found at {path}")
    return exists

def get_column_count_from_first_line(file_path: Path, logger: logging.Logger) -> int:
    """Get the number of columns from the first line of the CSV file.
    
    Args:
        file_path (Path): Path to the CSV file
        
    Returns:
        int: Number of columns in the CSV
        
    Raises:
        CSVDataError: If file is empty
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        validate_csv_not_empty(first_line, "Main CSV", logger)
        return len(first_line.split('\t'))

def get_default_sleep_stage_path(main_path: Union[str, Path]) -> str:
    """Get the default sleep stage path based on the main CSV path.
    
    This follows the convention of appending '.sleep.csv' to the main path.
    For example:
    - main_path: 'data/recording.csv' -> 'data/recording.sleep.csv'
    - main_path: 'data/recording' -> 'data/recording.sleep.csv'
    
    Args:
        main_path (Union[str, Path]): Path to the main CSV file
        
    Returns:
        str: The default sleep stage path
    """
    main_path = str(main_path)
    if main_path.endswith('.csv'):
        return main_path[:-4] + '.sleep.csv'
    return main_path + '.sleep.csv'

def clean_empty_sleep_stage_file(sleep_stage_path: Path) -> None:
    """Delete sleep stage file if it exists and is empty or contains only header."""
    if os.path.exists(sleep_stage_path):
        with open(sleep_stage_path, 'r') as f:
            content = f.read().strip()
            if content == "" or content == "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id":
                os.remove(sleep_stage_path)

def create_format_string(num_columns: int) -> str:
    """Create a tab-separated format string using the main data format specifier.
    
    Args:
        num_columns (int): Number of columns to create format string for
        
    Returns:
        str: Tab-separated format string
    """
    return '\t'.join([MAIN_DATA_FMT] * num_columns)

def create_format_specifiers(shape: int) -> List[str]:
    """Create a list of format specifiers for the given shape.
    
    Args:
        shape (int): Number of columns in the data
        
    Returns:
        List[str]: List of format specifiers, one for each column
    """
    return [MAIN_DATA_FMT] * shape

def create_column_names(num_columns: int, timestamp_channel: int) -> List[str]:
    """Create column names for the CSV DataFrame.
    
    Args:
        num_columns (int): Number of columns in the CSV
        timestamp_channel (int): Index of the timestamp channel
        
    Returns:
        List[str]: List of column names
    """
    column_names = [f'channel_{i}' for i in range(num_columns)]
    column_names[timestamp_channel] = 'timestamp'
    return column_names

def transform_data_to_rows(new_data: np.ndarray, logger=None) -> List[List[float]]:
    """Transform data from (channels, samples) format to (samples, channels) format for CSV writing.
    
    This transformation is necessary because:
    1. CSV files are naturally row-oriented (each row represents one time point)
    2. The final CSV format needs to be (samples, channels) for compatibility with other tools
    3. The transformation happens only once, at the final CSV writing stage
    
    Args:
        new_data (np.ndarray): Data in (n_channels, n_samples) format
            - Each row represents a channel
            - Each column represents a time point
            - Example: For 8 channels and 1000 samples: shape (8, 1000)
        logger (Optional[logging.Logger]): Logger instance for error reporting
        
    Returns:
        List[List[float]]: Data transformed to (samples, channels) format
            - Each inner list represents one time point
            - Each inner list contains values for all channels at that time point
            - Example: For 8 channels and 1000 samples:
                [
                    [sample1_ch1, sample1_ch2, ..., sample1_ch8],  # First time point
                    [sample2_ch1, sample2_ch2, ..., sample2_ch8],  # Second time point
                    ...
                    [sample1000_ch1, sample1000_ch2, ..., sample1000_ch8]  # Last time point
                ]
                
    Raises:
        CSVDataError: If transformed rows are empty
    """
    transformed_rows = new_data.T.tolist()
    validate_transformed_rows_not_empty(transformed_rows, logger)
    return transformed_rows

def filter_duplicate_timestamps(new_rows: List[List[float]], timestamp_channel: int, last_saved_timestamp: Optional[float], logger=None) -> Tuple[List[List[float]], int]:
    """Filter out rows with timestamps less than or equal to the last saved timestamp.
    
    Args:
        new_rows (List[List[float]]): List of data rows to filter
        timestamp_channel (int): Index of the timestamp channel in each row
        last_saved_timestamp (Optional[float]): Last saved timestamp to compare against
        logger (Optional[logging.Logger]): Logger instance for error reporting
        
    Returns:
        Tuple[List[List[float]], int]: Tuple containing:
            - Filtered rows (only rows with timestamps greater than last_saved_timestamp)
            - Number of duplicate rows that were filtered out
    """
    if last_saved_timestamp is None:
        return new_rows, 0
        
    timestamps = [row[timestamp_channel] for row in new_rows]
    duplicates = find_duplicates(timestamps, reference_value=last_saved_timestamp, comparison='less_equal') #TODO: figure out why we don't use this and if we ever use find_duplicates
    
    # Keep only rows whose timestamps are greater than last_saved_timestamp
    rows_to_add = [row for row in new_rows if row[timestamp_channel] > last_saved_timestamp]
    duplicate_count = len(new_rows) - len(rows_to_add)
    
    if logger and duplicate_count > 0:
        logger.debug(f"Skipped {duplicate_count} duplicate/overlapping samples from streaming")
        
    return rows_to_add, duplicate_count

def process_sleep_stage_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Process and validate sleep stage data from CSV.
    
    Args:
        file_path (Path): Path to the sleep stage CSV file
        
    Returns:
        Optional[pd.DataFrame]: Processed DataFrame or None if empty
        
    Raises:
        CSVFormatError: If data processing fails
    """
    # Read data
    sleep_stage_df = pd.read_csv(file_path, delimiter='\t')
        
    # Add string timestamp column for merging
    sleep_stage_df['timestamp_end_str'] = sleep_stage_df['timestamp_end'].astype(str)
    
    # Validate and convert timestamps
    sleep_stage_df = validate_sleep_stage_timestamps(sleep_stage_df)
    
    return sleep_stage_df.sort_values('timestamp_start')

def finalize_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Finalize the merged DataFrame by ensuring proper types and sorting.
    
    Args:
        merged_df (pd.DataFrame): DataFrame to finalize
        
    Returns:
        pd.DataFrame: Finalized DataFrame with proper types and sorting
        
    Raises:
        CSVFormatError: If timestamp conversion fails
    """
    try:
        merged_df['timestamp'] = pd.to_numeric(merged_df['timestamp'], errors='raise')
    except ValueError as e:
        raise CSVFormatError(f"Invalid timestamp format in main CSV: {e}")
    
    return merged_df.sort_values('timestamp')



 