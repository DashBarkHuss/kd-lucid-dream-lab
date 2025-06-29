"""
Validation module for CSV data export and validation of brainflow data.

This module provides validation functions for:
- Data shape and content validation
- File path validation
- CSV format validation
- Buffer state validation
- Timestamp continuity validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import logging
import os
from collections import Counter
from .exceptions import (
    CSVExportError, CSVValidationError, CSVDataError, CSVFormatError,
    MissingOutputPathError, BufferError, BufferOverflowError,
    BufferStateError, BufferValidationError
)

logger = logging.getLogger(__name__)

def validate_data_shape(data: np.ndarray) -> None:
    """Validate the shape and content of input data.
    
    Validation rules:
    - Input data must be 2D numpy array
    - No NaN or infinite values allowed in input data
    - After validation, sleep stage and buffer ID columns are added with NaN values
    - Timestamps must be monotonically increasing
    
    Args:
        data (np.ndarray): Data to validate
        
    Raises:
        CSVDataError: If data validation fails
    """
    if not isinstance(data, np.ndarray):
        raise CSVDataError("Input data must be a numpy array")
    
    if data.ndim != 2:
        raise CSVDataError(f"Input data must be 2-dimensional, got {data.ndim} dimensions")
    
    if data.size == 0:
        raise CSVDataError("Input data cannot be empty")
    
    # Check if data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise CSVDataError(f"Input data must be numeric, got dtype {data.dtype}")
    
    # Check for NaN and infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise CSVDataError("Data contains NaN or infinite values")

def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate and normalize file path.
    
    Validation rules:
    - Directory must exist
    - Path must be valid
    
    Args:
        file_path (Union[str, Path]): Path to validate
        
    Returns:
        Path: Normalized path object
        
    Raises:
        CSVExportError: If path validation fails
    """
    try:
        path = Path(file_path)
        if not path.parent.exists():
            raise CSVExportError(f"Directory does not exist: {path.parent}")
        return path
    except Exception as e:
        raise CSVExportError(f"Invalid file path: {e}")

def validate_sleep_stage_data(sleep_stage: float, buffer_id: float, timestamp_start: float, timestamp_end: float) -> None:
    """Validate sleep stage data inputs.
    
    Validation rules:
    - All inputs must be numeric (int or float)
    - Timestamps must be valid numbers
    
    Args:
        sleep_stage (float): Sleep stage classification
        buffer_id (float): ID of the buffer
        timestamp_start (float): Start timestamp for the sleep stage
        timestamp_end (float): End timestamp for the sleep stage
        
    Raises:
        CSVDataError: If any input validation fails
    """
    if not isinstance(sleep_stage, (int, float)):
        raise CSVDataError(f"Sleep stage must be numeric, got {type(sleep_stage)}")
    if not isinstance(buffer_id, (int, float)):
        raise CSVDataError(f"Buffer ID must be numeric, got {type(buffer_id)}")
    if not isinstance(timestamp_start, (int, float)):
        raise CSVDataError(f"Timestamp start must be numeric, got {type(timestamp_start)}")
    if not isinstance(timestamp_end, (int, float)):
        raise CSVDataError(f"Timestamp end must be numeric, got {type(timestamp_end)}")

def validate_sleep_stage_csv_format(header: str, first_line: Optional[str] = None) -> None:
    """Validate sleep stage CSV format.
    
    Validation rules:
    - Header must contain all required columns
    - First data line must match header column count
    - Timestamp must have 6 decimal places
    
    Args:
        header (str): CSV header line
        first_line (Optional[str]): First data line to validate
        
    Raises:
        CSVFormatError: If format validation fails
    """
    # Validate header format
    header_cols = header.split('\t')
    required_sleep_cols = ['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id']
    if not all(col in header_cols for col in required_sleep_cols):
        raise CSVFormatError(f"Sleep stage CSV missing required columns: {required_sleep_cols}")
    
    # Validate first data line if provided
    if first_line:
        first_line_cols = first_line.split('\t')
        if len(first_line_cols) != len(header_cols):
            raise CSVFormatError(f"Data line has {len(first_line_cols)} columns but header has {len(header_cols)} columns")
        
        # Validate timestamp format
        timestamp_end_str = first_line_cols[1]  # timestamp_end is second column
        try:
            float(timestamp_end_str)  # Try to convert to float
            if '.' not in timestamp_end_str or len(timestamp_end_str.split('.')[-1]) != 6:
                raise CSVFormatError(f"Timestamp end {timestamp_end_str} does not have six decimal places")
        except ValueError:
            raise CSVFormatError(f"Invalid timestamp format: {timestamp_end_str}")

def find_duplicates(timestamps: List[float], reference_value: Optional[float] = None, comparison: str = 'exact') -> Dict[float, int]:
    """Find duplicate timestamps in a list.
    
    This function can find either:
    1. Exact duplicates within the list (when comparison='exact')
    2. Values that match a condition against a reference value (when comparison='less_equal')
    
    Args:
        timestamps (List[float]): List of timestamps to check
        reference_value (Optional[float]): Reference value to compare against. Only used if comparison='less_equal'
        comparison (str): Type of comparison to perform:
            - 'exact': Find timestamps that appear more than once in the list
            - 'less_equal': Find timestamps that are <= reference_value
            
    Returns:
        Dict[float, int]: Dictionary of duplicate timestamps and their counts
        
    Raises:
        ValueError: If comparison is 'less_equal' but no reference_value provided
    """
    if comparison == 'less_equal' and reference_value is None:
        raise ValueError("reference_value must be provided when comparison='less_equal'")
        
    duplicates = {}
    if comparison == 'exact':
        # Find exact duplicates using Counter
        counts = Counter(timestamps)
        duplicates = {ts: count for ts, count in counts.items() if count > 1}
    elif comparison == 'less_equal':
        # Find values <= reference_value
        duplicates = {ts: 1 for ts in timestamps if ts <= reference_value}
        
    return duplicates

def validate_timestamps_unique(timestamps: List[float], logger: Optional[logging.Logger] = None) -> None:
    """Validate that all timestamps in a list are unique.
    
    This function checks for duplicate timestamps in a list and raises an error if any are found.
    It also logs detailed information about any duplicates found.
    
    Validation rules:
    - All timestamps must be unique
    - No duplicate timestamps allowed
    
    Args:
        timestamps (List[float]): List of timestamps to validate
        logger (Optional[logging.Logger]): Logger instance for error reporting
        
    Raises:
        CSVDataError: If duplicate timestamps are found
    """
    duplicates = find_duplicates(timestamps, comparison='exact')
    if duplicates:
        error_msg = f"Found {len(duplicates)} duplicate timestamps in buffer before saving"
        
        if logger:
            logger.error(error_msg + "!")
            for ts, count in duplicates.items():
                logger.error(f"Timestamp {ts} appears {count} times")
        
        raise CSVDataError(error_msg)

def validate_buffer_size_and_path(data_size: int, buffer_size: int, output_path: Optional[str]) -> None:
    """Validate that we have an output path if data would exceed buffer size.
    
    Validation rules:
    - If data size would exceed buffer size, output path must be set
    
    Args:
        data_size (int): Size of the data to be added
        buffer_size (int): Maximum size of the buffer
        output_path (Optional[str]): Path where data will be saved
        
    Raises:
        MissingOutputPathError: If data would exceed buffer size and no output path is set
    """
    if data_size > buffer_size and not output_path:
        logger.error(f"Missing output path: Initial data size {data_size} exceeds buffer size limit {buffer_size}")
        raise MissingOutputPathError(f"Output path must be set before accepting data that exceeds buffer size limit {buffer_size}")

def validate_data_not_empty(data: np.ndarray) -> None:
    """Validate that the data array is not empty.
    
    Args:
        data (np.ndarray): Data array to validate
        
    Raises:
        CSVDataError: If data array is empty
    """
    if not data.size:
        raise CSVDataError("Received empty data array")

def validate_transformed_rows_not_empty(rows: list, logger=None) -> None:
    """Validate that the transformed data rows are not empty.
    
    Args:
        rows (list): List of data rows after transformation
        logger (Optional[logging.Logger]): Logger instance for error reporting
        
    Raises:
        CSVDataError: If transformed rows are empty
    """
    if not rows:
        if logger:
            logger.error("Data transformation resulted in empty rows")
        raise CSVDataError("Data transformation resulted in empty rows")

def validate_timestamp_state(is_initial: bool, last_saved_timestamp: float, logger) -> None:
    """Validate that the timestamp state is correct for initial/non-initial data.
    
    Args:
        is_initial (bool): Whether this is initial data
        last_saved_timestamp (float): The last saved timestamp
        logger: Logger instance for debug messages
        
    Raises:
        BufferStateError: If timestamp state is invalid
    """
    if not is_initial and last_saved_timestamp is None:
        logger.error("last_saved_timestamp is None for non-initial data")
        raise BufferStateError("last_saved_timestamp is None for non-initial data. This indicates improper usage or a corrupted state. The timestamp should have been set during initial data processing.")

def validate_brainflow_data(new_data: np.ndarray) -> None:
    """Validate the brainflow data format and content.
    
    Args:
        new_data (np.ndarray): Input data to validate
        
    Raises:
        CSVDataError: If data validation fails
    """
    # Validate input data first
    validate_data_not_empty(new_data)
        
    # Validate data shape matches expected BrainFlow format
    validate_data_shape(new_data) 

def validate_add_to_buffer_requirements(new_data: np.ndarray, is_initial: bool, buffer_size: int, 
                                    csv_path: Optional[str], last_saved_timestamp: Optional[float], 
                                    logger: logging.Logger) -> None:
    """Validate all requirements before adding data to buffer.
    
    Args:
        new_data (np.ndarray): Input data to validate
        is_initial (bool): Whether this is the initial data chunk
        buffer_size (int): Maximum buffer size
        csv_path (Optional[str]): Path to CSV file
        last_saved_timestamp (Optional[float]): Last saved timestamp
        logger (logging.Logger): Logger instance
        
    Raises:
        CSVDataError: If data validation fails
        BufferOverflowError: If adding data would exceed buffer size limit
        BufferStateError: If buffer state is invalid
    """
    # Validate brainflow data format
    validate_brainflow_data(new_data)
    
    # Validate buffer size and output path requirements
    validate_buffer_size_and_path(len(new_data.T), buffer_size, csv_path)
    
    # Validate timestamp state
    validate_timestamp_state(is_initial, last_saved_timestamp, logger) 

def validate_output_path_set(output_path: Optional[str], path_type: str = "output", custom_message: Optional[str] = None) -> None:
    """Validate that an output path is set.
    
    Validation rules:
    - Path must not be None or empty
    
    Args:
        output_path (Optional[str]): Path to validate
        path_type (str): Type of path for error message (e.g., "main CSV", "sleep stage"). Defaults to "output".
        custom_message (Optional[str]): Optional custom error message. If provided, this will be used instead of the default message.
        
    Raises:
        MissingOutputPathError: If path is not set
    """
    if not output_path:
        message = custom_message if custom_message else f"No {path_type} path set"
        raise MissingOutputPathError(message)

def validate_main_csv_columns(num_columns: int, timestamp_channel: int) -> None:
    """Validate the column count and timestamp channel index of the main CSV file.
    
    Args:
        num_columns (int): Number of columns in the CSV
        timestamp_channel (int): Index of the timestamp channel
        
    Raises:
        CSVFormatError: If timestamp channel index exceeds number of columns
    """
    if timestamp_channel >= num_columns:
        raise CSVFormatError(f"Timestamp channel index {timestamp_channel} exceeds number of columns {num_columns}")

def validate_sleep_stage_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and convert sleep stage timestamps to numeric format.
    
    Args:
        df (pd.DataFrame): DataFrame containing sleep stage data
        
    Returns:
        pd.DataFrame: DataFrame with validated and converted timestamps
        
    Raises:
        CSVFormatError: If timestamp conversion fails
    """
    try:
        df['timestamp_start'] = pd.to_numeric(df['timestamp_start'], errors='raise')
        df['timestamp_end'] = pd.to_numeric(df['timestamp_end'], errors='raise')
        return df
    except ValueError as e:
        raise CSVFormatError(f"Invalid timestamp format: {e}")

def validate_sleep_stage_format(file_path: Path) -> None:
    """Validate the format of the sleep stage CSV file.
    
    Args:
        file_path (Path): Path to the sleep stage CSV file
        
    Raises:
        CSVFormatError: If format is invalid
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        validate_csv_not_empty(header, "Sleep stage CSV", logger)
        first_line = f.readline().strip()
        validate_sleep_stage_csv_format(header, first_line if first_line else None)

def validate_board_shim_set(board_shim: Optional[object], logger: Optional[logging.Logger] = None) -> None:
    """Validate that board_shim is set.
    
    Args:
        board_shim (Optional[object]): Board shim instance to validate
        logger (Optional[logging.Logger]): Logger for error reporting
        
    Raises:
        CSVExportError: If board_shim is not set
    """
    if board_shim is None:
        if logger:
            logger.error("board_shim is not set; cannot determine timestamp channel index")
        raise CSVExportError("board_shim is not set; cannot determine timestamp channel index")

def validate_csv_not_empty(first_line: str, file_type: str = "CSV", logger: Optional[logging.Logger] = None) -> None:
    """Validate that a CSV file is not empty.
    
    Args:
        first_line (str): First line of the CSV file
        file_type (str): Type of CSV file for error message (e.g., "Main CSV", "Sleep stage CSV")
        logger (Optional[logging.Logger]): Logger for error reporting
        
    Raises:
        CSVDataError: If file is empty
    """
    if not first_line:
        if logger:
            logger.error(f"{file_type} file is empty")
        raise CSVDataError(f"{file_type} file is empty")

def validate_buffer_not_empty(buffer: list, buffer_type: str = "buffer", logger: Optional[logging.Logger] = None) -> None:
    """Validate that a buffer is not empty.
    
    Args:
        buffer (list): Buffer to validate
        buffer_type (str): Type of buffer for error message (e.g., "main CSV buffer", "sleep stage buffer")
        logger (Optional[logging.Logger]): Logger instance for error reporting
        
    Raises:
        CSVDataError: If buffer is empty
    """
    if not buffer:
        if logger:
            logger.error(f"{buffer_type} is empty")
        raise CSVDataError(f"{buffer_type} is empty")

def validate_no_sleep_stage_overwrites(matching_samples: pd.DataFrame, timestamp: str, logger: logging.Logger) -> None:
    """Validate that there are no sleep stage or buffer ID overwrites.
    
    Args:
        matching_samples (pd.DataFrame): DataFrame containing the samples to check
        timestamp (str): Timestamp string for error reporting
        logger (logging.Logger): Logger instance for error reporting
        
    Raises:
        CSVDataError: If attempting to overwrite existing non-NaN values
    """
    if not matching_samples['sleep_stage'].isna().all() or not matching_samples['buffer_id'].isna().all():
        logger.error(
            f"Attempting to overwrite non-NaN values at timestamp {timestamp}. "
            f"Current values - Sleep Stage: {matching_samples['sleep_stage'].iloc[0]}, "
            f"Buffer ID: {matching_samples['buffer_id'].iloc[0]}"
        )
        raise CSVDataError("Cannot overwrite existing sleep stage or buffer ID values")

def validate_matching_timestamps(matching_samples: pd.DataFrame, timestamp: str, logger: logging.Logger) -> None:
    """Validate that there are matching timestamps between datasets.
    
    Args:
        matching_samples (pd.DataFrame): DataFrame containing the samples to check
        timestamp (str): Timestamp string for error reporting
        logger (logging.Logger): Logger instance for error reporting
        
    Raises:
        CSVDataError: If no matching timestamps are found
    """
    if matching_samples.empty:
        error_msg = f"No matching timestamp found for sleep stage end timestamp {timestamp}"
        logger.error(f"{error_msg}. This is an error because every sleep stage end timestamp should have a matching sample.")
        raise CSVDataError(error_msg)