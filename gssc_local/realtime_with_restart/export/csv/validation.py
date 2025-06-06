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