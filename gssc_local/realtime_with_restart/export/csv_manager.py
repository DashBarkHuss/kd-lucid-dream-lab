"""
CSV Manager for handling data export and validation.

This module provides functionality for:
- Saving data to CSV files
- Validating CSV data format
- Managing sleep stage data
- Ensuring data integrity
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union, Dict
import os
from pathlib import Path
import pandas as pd
from brainflow.board_shim import BoardShim

class CSVExportError(Exception):
    """Base exception for CSV export errors."""
    pass

class CSVValidationError(CSVExportError):
    """Raised when CSV validation fails."""
    pass

class CSVDataError(CSVExportError):
    """Raised when there are issues with the data being saved."""
    pass

class CSVFormatError(CSVExportError):
    """Raised when the CSV format is incorrect."""
    pass

class CSVManager:
    """Manages CSV data export and validation.
    
    This class handles all CSV-related operations including:
    - Saving data to CSV files
    - Validating CSV format and content
    - Managing sleep stage data
    - Ensuring data integrity during export
    
    Attributes:
        saved_data (List[List[float]]): Buffer for data to be saved
        output_csv_path (Optional[str]): Path where CSV will be saved
        last_saved_timestamp (Optional[float]): Timestamp of last saved data
        board_shim: BrainFlow board shim instance
    """
    
    def __init__(self, board_shim=None):
        """Initialize CSVManager.
        
        Args:
            board_shim: Optional board shim instance to get channel count
            
    
        """
     
        self.saved_data: List[List[float]] = []
        self.output_csv_path: Optional[str] = None
        self.last_saved_timestamp: Optional[float] = None
       
        self.board_shim = board_shim
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
    def _validate_data_shape(self, data: np.ndarray) -> None:
        """Validate the shape and content of input data.
        
        Args:
            data (np.ndarray): Data to validate
            
        Raises:
            CSVDataError: If data validation fails
        """
        if not isinstance(data, np.ndarray):
            raise CSVDataError("Input data must be a numpy array")
        
        if data.ndim != 2:
            raise CSVDataError(f"Input data must be 2-dimensional, got {data.ndim} dimensions")
            
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise CSVDataError("Data contains NaN or infinite values")
    
    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and normalize file path.
        
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
    
    def _validate_sleep_stage_data(self, sleep_stage: float, next_buffer_id: float, epoch_end_idx: int) -> None:
        """Validate sleep stage data before adding to CSV.
        
        Args:
            sleep_stage (float): Sleep stage classification
            next_buffer_id (float): ID of the next buffer
            epoch_end_idx (int): Index where to add the data
            
        Raises:
            CSVDataError: If validation fails
        """
        if not isinstance(sleep_stage, (int, float)):
            raise CSVDataError(f"Sleep stage must be numeric, got {type(sleep_stage)}")
            
        if not isinstance(next_buffer_id, (int, float)):
            raise CSVDataError(f"Buffer ID must be numeric, got {type(next_buffer_id)}")
            
        if not isinstance(epoch_end_idx, int):
            raise CSVDataError(f"Epoch index must be integer, got {type(epoch_end_idx)}")
            
        if epoch_end_idx < 0 or epoch_end_idx >= len(self.saved_data):
            raise CSVDataError(f"Epoch index {epoch_end_idx} out of bounds for saved data length {len(self.saved_data)}")
    
    def save_new_data(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
        """Save new data to the buffer for later CSV export.
        
        Args:
            new_data (np.ndarray): New data to save (channels x samples)
            is_initial (bool): Whether this is the initial data chunk
            
        Returns:
            bool: True if data was saved successfully
            
        Raises:
            CSVDataError: If data validation fails
        """
        try:
            self.logger.info(f"\n=== CSVManager.save_new_data() ===")
            self.logger.info(f"Input data shape: {new_data.shape}")
            self.logger.info(f"Is initial: {is_initial}")
            self.logger.info(f"Current saved_data length: {len(self.saved_data)}")
            
            self._validate_data_shape(new_data)
            
            # Convert data to list format and add NaN placeholders for sleep stage and buffer ID
            new_rows = new_data.T.tolist()
            self.logger.info(f"Number of new rows: {len(new_rows)}")
            
            for row in new_rows:
                row.extend([float('nan'), float('nan')])  # Add NaN for sleep stage and buffer ID
            
            # For initial data, save everything
            if is_initial:
                self.logger.info("Handling initial data")
                self.saved_data = new_rows
                if new_rows:
                    self.last_saved_timestamp = new_rows[-1][self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())]
                    self.logger.info(f"Set last_saved_timestamp to: {self.last_saved_timestamp}")
                return True
            
            # For subsequent data, only filter out exact duplicates
            if self.last_saved_timestamp is not None:
                self.logger.info(f"Processing subsequent data with last_saved_timestamp: {self.last_saved_timestamp}")
                # Find the first row with a timestamp greater than the last saved timestamp
                start_idx = 0
                for i, row in enumerate(new_rows):
                    if row[self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())] > self.last_saved_timestamp:
                        start_idx = i
                        break
                self.logger.info(f"Found start_idx: {start_idx}")
                
                # Save all rows from that point forward
                if start_idx < len(new_rows):
                    rows_to_add = new_rows[start_idx:]
                    self.logger.info(f"Adding {len(rows_to_add)} new rows")
                    self.saved_data.extend(rows_to_add)
                    self.last_saved_timestamp = new_rows[-1][self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())]
                    self.logger.info(f"Updated last_saved_timestamp to: {self.last_saved_timestamp}")
                else:
                    self.logger.info("No new rows to add")
            else:
                self.logger.info("No last_saved_timestamp, saving all rows")
                # If no last saved timestamp, save all rows
                self.saved_data.extend(new_rows)
                if new_rows:
                    self.last_saved_timestamp = new_rows[-1][self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())]
                    self.logger.info(f"Set last_saved_timestamp to: {self.last_saved_timestamp}")
            
            self.logger.info(f"Final saved_data length: {len(self.saved_data)}")
            self.logger.info("=== End save_new_data() ===\n")
            return True
            
        except CSVDataError as e:
            self.logger.error(f"Failed to save new data: {e}")
            raise
    
    def save_to_csv(self, output_path: Union[str, Path]) -> bool:
        """Save buffered data to CSV file.
        
        Args:
            output_path (Union[str, Path]): Path where to save the CSV file
            
        Returns:
            bool: True if save was successful
            
        Raises:
            CSVExportError: If save operation fails
        """
        try:
            path = self._validate_file_path(output_path)
            
            if not self.saved_data:
                self.logger.warning("No data to save to CSV")
                raise CSVExportError("No data to save to CSV")
            
            # Convert to numpy array
            data_array = np.array(self.saved_data, dtype=float)
            
            # Create format specifiers - all columns use float format
            fmt = ['%.6f'] * data_array.shape[1]
            
            # Save with exact format matching
            np.savetxt(path, data_array, delimiter='\t', fmt=fmt)
            
            self.output_csv_path = str(path)
            self.logger.info(f"Successfully saved data to {path}")
            return True
            
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save CSV due to I/O error: {e}")
            raise CSVExportError(f"Failed to save CSV due to I/O error: {e}")
        except ValueError as e:
            self.logger.error(f"Failed to save CSV due to invalid data: {e}")
            raise CSVDataError(f"Failed to save CSV due to invalid data: {e}")
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {e}")
            raise CSVExportError(f"Failed to save CSV: {e}")
    
    def validate_saved_csv_format(self, original_csv_path: Union[str, Path]) -> bool:
        """Validate the format and structure of the saved CSV.
        
        Args:
            original_csv_path (Union[str, Path]): Path to original CSV for comparison
            
        Returns:
            bool: True if validation passes
            
        Raises:
            CSVValidationError: If validation fails
        """
        try:
            path = self._validate_file_path(original_csv_path)
            
            if not self.output_csv_path:
                raise CSVValidationError("No CSV has been saved yet")
            
            # Read both CSVs
            original_df = pd.read_csv(path, header=None)
            saved_df = pd.read_csv(self.output_csv_path, header=None)
            
            # Compare shapes
            if original_df.shape[1] != saved_df.shape[1]:
                raise CSVValidationError(
                    f"Column count mismatch: original has {original_df.shape[1]}, saved has {saved_df.shape[1]}"
                )
            
            # Compare data types
            if not all(original_df.dtypes == saved_df.dtypes):
                raise CSVValidationError("Data type mismatch between original and saved CSV")
            
            # Compare timestamp column for continuity
            timestamp_col = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
            if not self._validate_timestamp_continuity(saved_df[timestamp_col]):
                raise CSVValidationError("Timestamp continuity check failed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate CSV format: {e}")
            raise CSVValidationError(f"CSV format validation failed: {e}")
    
    def validate_saved_csv_matches_original_source(self, original_csv_path: Union[str, Path]) -> bool:
        """Validate that the saved CSV matches the original format exactly, ignoring sleep stage and buffer ID columns.
        
        Args:
            original_csv_path (Union[str, Path]): Path to original CSV for comparison
            
        Returns:
            bool: True if validation passes
            
        Raises:
            CSVValidationError: If validation fails
        """
        try:
            path = self._validate_file_path(original_csv_path)
            
            if not self.output_csv_path:
                raise CSVValidationError("No CSV has been saved yet")
            
            # Read both CSVs as strings first
            with open(self.output_csv_path, 'r') as f:
                saved_lines = f.readlines()
            with open(path, 'r') as f:
                original_lines = f.readlines()
            
            self.logger.info("\nCSV Validation Results:")
            
            # Check number of lines
            if len(saved_lines) != len(original_lines):
                self.logger.error(f"❌ Line count mismatch: Original={len(original_lines)}, Saved={len(saved_lines)}")
                return False
            self.logger.info(f"✅ Line count matches: {len(original_lines)} lines")
            
            # Compare each line exactly, but only the original columns
            for i, (saved_line, original_line) in enumerate(zip(saved_lines, original_lines)):
                # Split lines into columns and remove the last two columns from saved data
                saved_columns = saved_line.strip().split('\t')[:-2]  # Remove sleep stage and buffer ID
                original_columns = original_line.strip().split('\t')
                
                # Rejoin columns for comparison
                saved_line_trimmed = '\t'.join(saved_columns)
                
                if saved_line_trimmed != original_line.strip():
                    self.logger.error(f"❌ Line {i+1} does not match exactly:")
                    self.logger.error(f"Original: {original_line.strip()}")
                    self.logger.error(f"Saved:    {saved_line_trimmed}")
                    return False
            
            self.logger.info("✅ All lines match exactly (ignoring sleep stage and buffer ID columns)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate CSV against original source: {e}")
            raise CSVValidationError(f"CSV source validation failed: {e}")
    
    def _validate_timestamp_continuity(self, timestamps: pd.Series) -> bool:
        """Validate basic timestamp integrity.
        
        Only checks for:
        1. No NaN values
        2. Monotonic increasing timestamps
        
        Does not check for gaps since gaps are expected in the real data stream.
        The actual gap detection and handling is done during real-time processing.
        
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
    
    def add_sleep_stage_to_csv(self, sleep_stage: float, next_buffer_id: float, epoch_end_idx: int) -> None:
        """Add sleep stage and buffer ID to saved data.
        
        Args:
            sleep_stage (float): Sleep stage classification
            next_buffer_id (float): ID of the next buffer
            epoch_end_idx (int): Index where to add the data
            
        Raises:
            CSVDataError: If validation fails
        """
        try:
            self._validate_sleep_stage_data(sleep_stage, next_buffer_id, epoch_end_idx)
            
            # Update the last two columns (which were initialized as NaN)
            self.saved_data[epoch_end_idx][-2] = float(sleep_stage)  # Convert to float
            self.saved_data[epoch_end_idx][-1] = float(next_buffer_id)  # Convert to float
            
        except CSVDataError as e:
            self.logger.error(f"Failed to add sleep stage data: {e}")
            raise 