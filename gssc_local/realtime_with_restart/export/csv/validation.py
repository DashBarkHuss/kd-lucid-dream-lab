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

def validate_timestamp_continuity(self, timestamps: pd.Series) -> bool:
    """Validate basic timestamp integrity.
    
    Only checks for:
    1. No NaN values
    2. Monotonic increasing timestamps
    
    Does not check for gaps since gaps are expected in the real data stream.
    The actual gap detection and handling is done during real-time processing and epoch processing.
    
    Args:
        timestamps (pd.Series): Series of timestamps to validate
        
    Returns:
        bool: True if timestamps pass basic integrity checks
    """
    # Check for NaN values
    if timestamps.isna().any():
        return False
    
    # Check for monotonic increase
    if not timestamps.is_monotonic_increasing:
        return False
    
    return True

def validate_buffer_state(self) -> None:
    """Validate current buffer state.
    
    Checks if buffer is in a valid state for operations.
    
    Raises:
        BufferStateError: If buffer is in an invalid state
    """
    # Check if main buffer exists
    if not hasattr(self, 'main_csv_buffer'):
        raise BufferStateError("Main buffer not initialized")
        
    # Check if sleep stage buffer exists
    if not hasattr(self, 'sleep_stage_buffer'):
        raise BufferStateError("Sleep stage buffer not initialized")
        
    # Check if buffer size is set
    if not hasattr(self, 'main_buffer_size'):
        raise BufferStateError("Main buffer size not configured")
        
    # Check if sleep stage buffer size is set
    if not hasattr(self, 'sleep_stage_buffer_size'):
        raise BufferStateError("Sleep stage buffer size not configured")
    
def validate_buffer_data(self, data: List[List[float]]) -> None:
    """Validate general buffer data integrity.
    
    This is a general-purpose validation that checks:
    - Data structure (list of lists)
    - Consistent row lengths
    - Numeric values
    - No NaN or infinite values
    
    Note: This is not BrainFlow-specific validation. For BrainFlow data,
    additional validation should be performed using BrainFlow utilities.
    
    Args:
        data (List[List[float]]): Data to validate
        
    Raises:
        BufferValidationError: If data validation fails
    """
    if not isinstance(data, list):
        raise BufferValidationError(f"Data must be a list, got {type(data)}")
        
    if not data:
        return  # Empty data is valid
        
    # Check if all rows have the same length
    row_length = len(data[0])
    for i, row in enumerate(data):
        if not isinstance(row, list):
            raise BufferValidationError(f"Row {i} must be a list, got {type(row)}")
        if len(row) != row_length:
            raise BufferValidationError(f"Row {i} has length {len(row)}, expected {row_length}")
            
    # Check if all values are numeric
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                raise BufferValidationError(f"Value at row {i}, column {j} must be numeric, got {type(value)}")
            if np.isnan(value) or np.isinf(value):
                raise BufferValidationError(f"Value at row {i}, column {j} is NaN or infinite")

def validate_file_contents(self, file_path: str) -> Dict:
    """Validate file contents and track duplicates.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        Dict containing validation results:
        {
            'total_lines': int,
            'unique_timestamps': set,
            'duplicate_timestamps': set,
            'timestamp_ranges': List[Tuple[float, float]],
            'gaps': List[Tuple[float, float]]
        }
    """
    try:
        if not os.path.exists(file_path):
            return {
                'total_lines': 0,
                'unique_timestamps': set(),
                'duplicate_timestamps': set(),
                'timestamp_ranges': [],
                'gaps': []
            }
        
        # Get timestamp channel index
        timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
        
        # Read file and track timestamps
        timestamps = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    values = line.strip().split('\t')
                    if len(values) > timestamp_channel:
                        try:
                            ts = float(values[timestamp_channel])
                            timestamps.append(ts)
                        except (ValueError, IndexError):
                            continue
        
        # Find duplicates
        unique_timestamps = set(timestamps)
        duplicate_timestamps = set(ts for ts in timestamps if timestamps.count(ts) > 1)
        
        # Find timestamp ranges and gaps
        if timestamps:
            sorted_timestamps = sorted(timestamps)
            timestamp_ranges = []
            gaps = []
            
            # Calculate expected time between samples (assuming uniform sampling)
            if len(sorted_timestamps) > 1:
                time_diffs = [sorted_timestamps[i+1] - sorted_timestamps[i] for i in range(len(sorted_timestamps)-1)]
                median_diff = sorted(time_diffs)[len(time_diffs)//2]
                gap_threshold = median_diff * 2  # Consider gaps larger than 2x median as actual gaps
                
                # Find ranges and gaps
                start = sorted_timestamps[0]
                for i in range(1, len(sorted_timestamps)):
                    if sorted_timestamps[i] - sorted_timestamps[i-1] > gap_threshold:
                        timestamp_ranges.append((start, sorted_timestamps[i-1]))
                        gaps.append((sorted_timestamps[i-1], sorted_timestamps[i]))
                        start = sorted_timestamps[i]
                timestamp_ranges.append((start, sorted_timestamps[-1]))
        
        return {
            'total_lines': len(timestamps),
            'unique_timestamps': unique_timestamps,
            'duplicate_timestamps': duplicate_timestamps,
            'timestamp_ranges': timestamp_ranges if timestamps else [],
            'gaps': gaps if timestamps else []
        }
        
    except Exception as e:
        self.logger.error(f"Failed to validate file contents: {e}")
        raise CSVValidationError(f"Failed to validate file contents: {e}")

def validate_saved_csv_matches_original_source(self, original_csv_path: Union[str, Path], output_path: Union[str, Path] = None) -> bool:
    """Validate that the saved CSV matches the original format exactly.
    
    Note: This is a test function and not part of real-time processing.
    TODO: This validation function should be moved to a test as it is not relevant to real-time processing.
    
    Validation rules:
    - Line count must match original
    - Each line must match exactly
    
    Args:
        original_csv_path (Union[str, Path]): Path to original CSV for comparison
        output_path (Union[str, Path], optional): Path to output CSV for comparison. 
            Defaults to self.main_csv_path if not provided.
        
    Returns:
        bool: True if validation passes
        
    Raises:
        CSVValidationError: If validation fails
    """
    try:
        path = validate_file_path(original_csv_path)

        if output_path is None:
            output_path = self.main_csv_path
        
        # Read both CSVs as strings first
        with open(output_path, 'r') as f:
            saved_lines = f.readlines()
        with open(path, 'r') as f:
            original_lines = f.readlines()
        
        self.logger.info("\nCSV Validation Results:")
        self.logger.info(f"Original CSV path: {path}")
        self.logger.info(f"Saved CSV path: {output_path}")
        self.logger.info(f"Original lines: {len(original_lines)}")
        self.logger.info(f"Saved lines: {len(saved_lines)}")
        
        # Check number of lines
        if len(saved_lines) != len(original_lines):
            self.logger.error(f"❌ Line count mismatch: Original={len(original_lines)}, Saved={len(saved_lines)}")
            # Debug first few lines of both files
            self.logger.error("\nFirst 5 lines of original file:")
            for i, line in enumerate(original_lines[:5]):
                self.logger.error(f"Original line {i+1}: {line.strip()}")
            self.logger.error("\nFirst 5 lines of saved file:")
            for i, line in enumerate(saved_lines[:5]):
                self.logger.error(f"Saved line {i+1}: {line.strip()}")
            raise CSVValidationError(f"Line count mismatch: Original={len(original_lines)}, Saved={len(saved_lines)}")
        if len(saved_lines) == 0 and len(original_lines) == 0:
            self.logger.error("❌ Both saved and reference CSV files are empty.")
            raise CSVValidationError("Both saved and reference CSV files are empty.")
        self.logger.info(f"✅ Line count matches: {len(original_lines)} lines")
        
        # Compare each line exactly
        for i, (saved_line, original_line) in enumerate(zip(saved_lines, original_lines)):
            if saved_line.strip() != original_line.strip():
                self.logger.error(f"❌ Line {i+1} does not match exactly:")
                self.logger.error(f"Original: {original_line.strip()}")
                self.logger.error(f"Saved:    {saved_line.strip()}")
                # Debug surrounding lines for context
                start = max(0, i-2)
                end = min(len(saved_lines), i+3)
                self.logger.error("\nContext from original file:")
                for j in range(start, end):
                    self.logger.error(f"Original line {j+1}: {original_lines[j].strip()}")
                self.logger.error("\nContext from saved file:")
                for j in range(start, end):
                    self.logger.error(f"Saved line {j+1}: {saved_lines[j].strip()}")
                raise CSVValidationError(f"Line {i+1} does not match exactly")
        
        self.logger.info("✅ All lines match exactly")
        return True
        
    except Exception as e:
        self.logger.error(f"Failed to validate CSV against original source: {e}")
        raise CSVValidationError(f"CSV source validation failed: {e}") 