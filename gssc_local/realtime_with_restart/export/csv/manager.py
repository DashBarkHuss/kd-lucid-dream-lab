"""
CSV Manager for handling data export and validation of brainflow data. 
TODO: refactor- this is doing repetitive tasks like saving two different buffer sizes for two different csv's (sleep staging and main eeg). we probably want an brianflow specific orchestrator and a csv manager class.

This module provides functionality for saving and validating data from brainflow streaming to a csv file with sleep stage integration.
It implements a memory-efficient buffer management system that prevents memory overflow during long recordings.

Key Features:
- Memory-efficient buffer management with configurable buffer sizes
- Incremental saving to prevent memory overflow
- Separate handling of main EEG data and sleep stage data
- Exact format preservation for compatibility
- Comprehensive validation and error handling


See individual method docstrings for detailed documentation.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union, Dict
import os
from pathlib import Path
import pandas as pd
from .exceptions import (
    CSVExportError, CSVValidationError, CSVDataError, CSVFormatError,
    MissingOutputPathError, BufferError, BufferOverflowError,
    BufferStateError, BufferValidationError
)
from .validation import (
    validate_data_shape,
    validate_file_path,
    validate_saved_csv_matches_original_source,
    validate_sleep_stage_data,
    validate_sleep_stage_csv_format,
    validate_buffer_size_and_path,
    validate_timestamps_unique,
    validate_data_not_empty,
    validate_transformed_rows_not_empty,
    validate_timestamp_state,
    validate_brainflow_data,
    validate_add_to_buffer_requirements
)

class CSVManager:
    """Manages CSV data export and validation for BrainFlow data.
    
    Key Features:
    - Memory-efficient buffer management with incremental saving to prevent overflow
    - Saving BrainFlow data to CSV files with exact format preservation
    - Managing sleep stage data integration with BrainFlow data
    - Testing CSV format against original BrainFlow source (test only) TODO: Move to a test
    
    See individual method docstrings for detailed documentation.
    
    Attributes:
        main_csv_buffer (List[List[float]]): Buffer for BrainFlow data to be saved
        main_csv_path (Optional[str]): Path where main BrainFlow data CSV will be saved
        sleep_stage_csv_path (Optional[str]): Path where sleep stage CSV will be saved
        last_saved_timestamp (Optional[float]): Timestamp of last saved data row/sample
        board_shim: BrainFlow board shim instance for channel configuration
        logger: Logger instance for error reporting and debugging
        main_buffer_size (int): Maximum number of samples to keep in main buffer (default: 10,000)
        sleep_stage_buffer (List[Tuple[float, float, float, float]]): Buffer for sleep stage data
        sleep_stage_buffer_size (int): Maximum number of entries in sleep stage buffer (default: 100)
    """
    
    def __init__(self, board_shim=None, main_buffer_size: int = 10_000, 
                 sleep_stage_buffer_size: int = 100, main_csv_path: Optional[str] = None,
                 sleep_stage_csv_path: Optional[str] = None):
        """Initialize CSVManager.
        
        Args:
            board_shim: Optional board shim instance to get channel count
            main_buffer_size (int): Maximum number of samples to keep in main buffer (default: 10,000)
            sleep_stage_buffer_size (int): Maximum number of entries in sleep stage buffer (default: 100)
            main_csv_path (Optional[str]): Path where main BrainFlow data CSV will be saved
            sleep_stage_csv_path (Optional[str]): Path where sleep stage CSV will be saved
        """
        # Initialize basic attributes
        self.main_csv_buffer: List[List[float]] = []
        self.main_csv_path: Optional[str] = main_csv_path
        self.sleep_stage_csv_path: Optional[str] = sleep_stage_csv_path
        self.last_saved_timestamp: Optional[float] = None
        self.board_shim = board_shim
        self.logger = logging.getLogger(__name__)
        
        # Set buffer sizes first
        self.main_buffer_size: int = main_buffer_size
        self.sleep_stage_buffer_size: int = sleep_stage_buffer_size
        
        # Initialize other attributes
        self.sleep_stage_buffer: List[Tuple[float, float, float, float]] = []  # (timestamp_start, timestamp_end, sleep_stage, buffer_id)
        
        # Create bound validation methods # TODO: some of these methods are never used outside of the tests
        self._validate_file_path = lambda path: validate_file_path(path)
        self.validate_saved_csv_matches_original_source = lambda original_csv_path, output_path=None: validate_saved_csv_matches_original_source(self, original_csv_path, output_path)
    
   
    def _check_buffer_overflow(self, buffer: List[Tuple[float, float, float, float]], buffer_size: int) -> bool:
        """Check if buffer would overflow with current size.

        Returns:
            bool: True if buffer would overflow, False otherwise
        """
        return len(buffer) >= buffer_size
    
    def _check_main_buffer_overflow(self) -> bool:
        """Check if main buffer (BrainFlow data) would overflow with current size.
        
        Returns:
            bool: True if main buffer would overflow, False otherwise
        """
        return self._check_buffer_overflow(self.main_csv_buffer, self.main_buffer_size)
    
    def _check_sleep_stage_buffer_overflow(self) -> bool:
        """Check if sleep stage buffer would overflow with current size.
        
        Returns:
            bool: True if sleep stage buffer would overflow, False otherwise
        """
        return self._check_buffer_overflow(self.sleep_stage_buffer, self.sleep_stage_buffer_size)


    def clear_output_file(self) -> None: # TODO: naming is confusing becuase it doens't except a file path.
        """Clear both the main CSV file and sleep stage CSV file if they exist."""
        if self.main_csv_path and os.path.exists(self.main_csv_path):
            self.logger.debug("Clearing existing main output file for new run.")
            with open(self.main_csv_path, 'w') as f:
                pass
        if self.sleep_stage_csv_path and os.path.exists(self.sleep_stage_csv_path):
            self.logger.debug("Clearing existing sleep stage output file for new run.")
            with open(self.sleep_stage_csv_path, 'w') as f:
                pass

    def _filter_duplicate_timestamps(self, new_rows: List[List[float]], timestamp_channel: int) -> Tuple[List[List[float]], int]:
        """Filter out rows with timestamps less than or equal to the last saved timestamp.
        
        Args:
            new_rows (List[List[float]]): List of data rows to filter
            timestamp_channel (int): Index of the timestamp channel in each row
            
        Returns:
            Tuple[List[List[float]], int]: Tuple containing:
                - Filtered rows (only rows with timestamps greater than last_saved_timestamp)
                - Number of duplicate rows that were filtered out
        """
        start_idx = 0
        duplicate_count = 0
        
        for i, row in enumerate(new_rows):
            if row[timestamp_channel] <= self.last_saved_timestamp:
                duplicate_count += 1
            else:
                start_idx = i
                break
        
        rows_to_add = new_rows[start_idx:]
        return rows_to_add, duplicate_count

    def _update_last_saved_timestamp(self, timestamp_channel: int) -> None:
        """Update the last saved timestamp from the current buffer if it has data.
        
        Args:
            timestamp_channel (int): Index of the timestamp channel in each row
        """
        if self.main_csv_buffer:
            self.last_saved_timestamp = self.main_csv_buffer[-1][timestamp_channel]

    def _handle_row_addition(self, new_rows: List[List[float]], timestamp_channel: int, is_initial: bool) -> None:
        """Handle the addition of rows to the buffer based on whether it's initial or subsequent data.
        
        Args:
            new_rows (List[List[float]]): The rows to be added
            timestamp_channel (int): Index of the timestamp channel
            is_initial (bool): Whether this is the initial data chunk
        """
        if is_initial:
            # Clear the output file early for initial data
            self.clear_output_file()
            # Add all rows for initial data first
            self.main_csv_buffer.extend(new_rows)
        else:
            # For subsequent data, handle duplicates by finding the first new timestamp
            rows_to_add, duplicate_count = self._filter_duplicate_timestamps(new_rows, timestamp_channel)
            
            if duplicate_count > 0:
                self.logger.debug(f"Skipped {duplicate_count} duplicate/overlapping samples from streaming")
            
            # Add filtered rows to buffer
            if rows_to_add:
                self.main_csv_buffer.extend(rows_to_add)
            else:
                self.logger.debug("No new samples to add after filtering duplicates")

    def _transform_data_to_rows(self, new_data: np.ndarray) -> List[List[float]]:
        """Transform numpy array data into a list of rows.
        
        Args:
            new_data (np.ndarray): Input data in channels x samples format
            
        Returns:
            List[List[float]]: Data transformed into list of rows format
            
        Raises:
            CSVDataError: If transformed rows are empty
        """
        transformed_rows = new_data.T.tolist()
        validate_transformed_rows_not_empty(transformed_rows, self.logger)
        return transformed_rows

    def add_data_to_buffer(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
        """Add new data to the buffer and handle buffer management.

        Handles continuous streaming data that needs careful timestamp management and duplicate prevention.
        
        This method handles:
        - Data validation
        - Managing possible duplicate timestamps from stream
        - Buffer size checks
        - Triggering saves when buffer is full
        - Initial vs subsequent data handling
        - Buffer clearing when necessary
        
        Args:
            new_data (np.ndarray): New data to add (channels x samples). In practice, this comes from raw brainflow data.
            is_initial (bool): Whether this is the initial data chunk
            
        Returns:
            bool: True if data was added successfully
            
        Raises:
            CSVDataError: If data validation fails
            BufferOverflowError: If adding data would exceed buffer size limit
        """
        try:
            # Validate all requirements before processing
            validate_add_to_buffer_requirements(new_data, is_initial, self.main_buffer_size,
                                             self.main_csv_path, self.last_saved_timestamp, self.logger)

            # Convert data to list of rows, important to transpose the brainflow data first
            transformed_rows = self._transform_data_to_rows(new_data)

            # Get timestamp channel index
            timestamp_channel = self._get_timestamp_channel_index()

            # Handle row addition
            self._handle_row_addition(transformed_rows, timestamp_channel, is_initial)

            # Update last saved timestamp if buffer has data
            self._update_last_saved_timestamp(timestamp_channel)
                
            # Then check if buffer size exceeds limit
            if self._check_main_buffer_overflow():
                # Save current buffer
                self.save_main_buffer_to_csv(is_initial=is_initial)  # This clears the buffer

            return True
            
        except CSVDataError as e:
            self.logger.error(f"Failed to add data to buffer: {e}")
            raise
        except BufferOverflowError as e:
            self.logger.error(f"Buffer overflow: {e}")
            raise

    def save_all_data(self) -> bool:
        """Save all remaining data in both main and sleep stage buffers to their respective CSV files.
        
        This method handles:
        1. Saving any remaining data in the main buffer to the main CSV
        2. Saving any remaining sleep stage data to the sleep stage CSV
        
        Returns:
            bool: True if all data was saved successfully or if there was no data to save
            
        Raises:
            CSVExportError: If save operation fails or if paths are not configured
            CSVDataError: If data validation fails
        """
        try:

            # If no data to save, just return success
            if not self.main_csv_buffer and not self.sleep_stage_buffer:
                return True
            
            # Only check paths if we have data to save
            if not self.main_csv_path:
                raise CSVExportError("Main CSV path not configured")
            if not self.sleep_stage_csv_path:
                raise CSVExportError("Sleep stage CSV path not configured")
            

            # Save any remaining data in the main buffer
            if self.main_csv_buffer:
                self.save_main_buffer_to_csv()
                
            # Save any remaining sleep stage data
            if self.sleep_stage_buffer:
                self.save_sleep_stages_to_csv()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save all data: {e}")
            raise CSVExportError(f"Failed to save all data: {e}")

    def _get_default_sleep_stage_path(self, main_path: Union[str, Path]) -> str:
        """Get the default sleep stage path based on the main CSV path.
        
        This is a convenience method that follows the convention of appending '.sleep.csv'
        to the main path. For example:
        - main_path: 'data/recording.csv'
        - returns: 'data/recording.sleep.csv'
        
        Args:
            main_path (Union[str, Path]): Path to the main CSV file
            
        Returns:
            str: The default sleep stage path
        """
        main_path = str(main_path)
        if main_path.endswith('.csv'):
            return main_path[:-4] + '.sleep.csv'
        return main_path + '.sleep.csv'

    def save_all_and_cleanup(self, output_path: Optional[Union[str, Path]] = None, merge_files: bool = False, merge_output_path: Optional[Union[str, Path]] = None) -> bool:
        """Save all remaining data and perform cleanup.
        
        This is a convenience method that combines save_all_data() and cleanup().
        It saves all remaining data in both buffers and then performs cleanup.
        
        Args:
            output_path (Optional[Union[str, Path]]): Path where to save the main CSV file.
                If not provided, uses the existing main_csv_path.
            
        Returns:
            bool: True if save and cleanup were successful
            
        Raises:
            CSVExportError: If save operation fails
            CSVDataError: If data validation fails
        """
        try:
            self.logger.info(f"\n=== CSVManager.save_all_and_cleanup() ===")
            # Use existing output path if none provided
            path_to_use = output_path if output_path is not None else self.main_csv_path
            if path_to_use is None:
                raise CSVExportError("No main CSV path provided and no existing main_csv_path set")
            
            # Set main path
            self.main_csv_path = str(path_to_use)
            
            # If sleep stage path not set, use convention
            if self.sleep_stage_csv_path is None:
                self.sleep_stage_csv_path = self._get_default_sleep_stage_path(path_to_use)
            
            result = self.save_all_data()

            if merge_files:
                self.merge_files(self.main_csv_path, self.sleep_stage_csv_path, merge_output_path)

            self.cleanup(reset_paths=True)
            self.logger.info("=== End save_all_and_cleanup() ===\n")
            return result
        except Exception as e:
            self.logger.error(f"Failed to save all data and cleanup: {e}")
            raise CSVExportError(f"Failed to save all data and cleanup: {e}")



    
    def add_sleep_stage_to_sleep_stage_csv(self, sleep_stage: float, buffer_id: float, timestamp_start: float, timestamp_end: float) -> bool:
        """Add sleep stage data to the sleep stage buffer.

        Handles discrete events (sleep stage classifications) that are inherently unique and don't need the same level of processing as continuous data, such as brainflow data.
        
        This method handles:
        - Adding sleep stage data to the sleep stage buffer
        - Buffer size limit checks
        - Triggering save when buffer is full
        
        Args:
            sleep_stage (float): Sleep stage classification
            buffer_id (float): ID of the buffer
            timestamp_start (float): Start timestamp for the sleep stage
            timestamp_end (float): End timestamp for the sleep stage
            
        Returns:
            bool: True if data was added successfully
            
        Raises:
            CSVDataError: If validation fails
            CSVExportError: If save operation fails
            MissingOutputPathError: If buffer is full and no output path is set
        """
        try:
            # Validate inputs using the new validation function
            validate_sleep_stage_data(sleep_stage, buffer_id, timestamp_start, timestamp_end)
            
            # Check if adding this entry would exceed buffer size
            if self._check_sleep_stage_buffer_overflow():
                self.save_sleep_stages_to_csv()  # This will handle path validation and errors
            
            # Prepare and add the new entry
            sleep_stage_entry = (float(timestamp_start), float(timestamp_end), float(sleep_stage), float(buffer_id))
            self.sleep_stage_buffer.append(sleep_stage_entry)
            
            return True
            
        except MissingOutputPathError:
            raise  # Re-raise MissingOutputPathError without wrapping
        except Exception as e:
            self.logger.error(f"Failed to add sleep stage to buffer: {e}")
            raise CSVDataError(f"Failed to add sleep stage to buffer: {e}")

    def _reset_last_saved_timestamp(self) -> None:
        """Reset the last saved timestamp to None."""
        self.last_saved_timestamp = None

    def cleanup(self, reset_paths: bool = True) -> None:
        """Clean up resources and reset state.
        
        This method should be called when the CSVManager is no longer needed
        to ensure proper resource release and state reset.
        
        Args:
            reset_paths (bool): Whether to reset output paths. Defaults to True.
                Set to False if you want to keep the paths for future saves.
                
        Raises:
            CSVExportError: If cleanup fails
        """
        try:
            # Clear data buffers
            self.main_csv_buffer.clear()
            self.sleep_stage_buffer.clear()
            
            # Reset state
            self._reset_last_saved_timestamp()
            
            # Optionally reset paths
            if reset_paths:
                self.main_csv_path = None
                self.sleep_stage_csv_path = None
            
        except Exception as e:
            self.logger.error(f"Error during CSVManager cleanup: {e}")
            raise CSVExportError(f"Failed to cleanup CSVManager: {e}")

    def save_main_buffer_to_csv(self, is_initial: bool = False) -> bool:
        """Save current main BrainFlow data buffer contents to CSV file and clear the buffer.
        
        This method handles both first write (create new file) and subsequent writes (append).
        It preserves the exact CSV format and clears the buffer after successful write.
        
        Returns:
            bool: True if save was successful or if there was no data to save
            
        Raises:
            CSVExportError: If save operation fails
            CSVDataError: If data validation fails
        """
        try:

            

                
            # TODO: This code block was causing data loss by deleting the CSV file when buffer was empty.
            # It's unclear what the original purpose was, and it seems to be a remnant from an earlier version.
            # Commenting out to prevent accidental data deletion.
            # if not self.main_csv_buffer:
            #     # If file exists and is empty, delete it
            #     if self.main_csv_path and os.path.exists(self.main_csv_path):
            #         os.remove(self.main_csv_path)
            #     return True # TODO: why is this returning True?
            
            if not self.main_csv_path:
                raise CSVExportError("No main CSV path set")
                
            # return early if buffer is empty
            if not self.main_csv_buffer:
                return True
            
            # Get timestamp channel index
            if self.board_shim is not None:
                timestamp_channel = self._get_timestamp_channel_index()
            else:
                raise CSVExportError("board_shim is not set; cannot determine timestamp channel index")

            # Validate timestamps in buffer before saving
            buffer_timestamps = [row[timestamp_channel] for row in self.main_csv_buffer]
            validate_timestamps_unique(buffer_timestamps, self.logger)
            
            # Convert to numpy array
            data_array = np.array(self.main_csv_buffer, dtype=float)
            
            # Create format specifiers - all columns use float format
            fmt = ['%.6f'] * data_array.shape[1]
            
            # Save with exact format matching
            if os.path.exists(self.main_csv_path):
                # Log before appending
                with open(self.main_csv_path, 'r') as f:
                    samples_before = sum(1 for _ in f)
                
                # Append to existing file
                with open(self.main_csv_path, 'a') as f:
                    np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
                
                # Log after appending
                with open(self.main_csv_path, 'r') as f:
                    samples_after = sum(1 for _ in f)
            else:
                # Log before creating new file
                 # TODO: this is too coupled. WE should be creating the file in the add_data_to_buffer method on the initial data chunk.
                
                # Create a new file
                np.savetxt(self.main_csv_path, data_array, delimiter='\t', fmt=fmt)
                
                # Log after creating new file
                with open(self.main_csv_path, 'r') as f:
                    samples_after = sum(1 for _ in f)
            
            # Update last saved timestamp before clearing buffer
            self._update_last_saved_timestamp(timestamp_channel)
            
            # Clear buffer after saving
            self.main_csv_buffer.clear()

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

    def _create_sleep_stage_file(self, data_array: Optional[np.ndarray] = None) -> None:
        """Create a new sleep stage file with header and optionally write data.
        
        Args:
            data_array (Optional[np.ndarray]): Data to write to the file. If None, only creates header.
            
        Raises:
            CSVExportError: If file creation fails
        """
        try:
            # Only create file if we have data to write
            if data_array is not None:
                header = "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n"
                with open(self.sleep_stage_csv_path, 'w') as f:
                    f.write(header)
                    # Create format specifiers:
                    # - timestamps: match BrainFlow's 6 decimal places for exact matching
                    # - sleep stage and buffer ID: use integer format since they're discrete values
                    fmt = ['%.6f', '%.6f', '%.0f', '%.0f']
                    np.savetxt(f, data_array, delimiter='\t', fmt=fmt)

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to create sleep stage file: {e}")
            raise CSVExportError(f"Failed to create sleep stage file: {e}")

    def save_sleep_stages_to_csv(self) -> bool:
        """Save current sleep stage buffer contents to CSV file and clear the buffer.
        
        This method handles both first write (create new file) and subsequent writes (append).
        For first write, it creates a new file with the required header:
        timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id
        
        For subsequent writes, it appends to the existing file.
        All numeric values are saved with 6 decimal places for consistency.
        
        Returns:
            bool: True if save was successful
            
        Raises:
            CSVExportError: If save operation fails
            CSVDataError: If data validation fails
            MissingOutputPathError: If no output path is set
        """
        try:
            # First check if we have any data to save
            if not self.sleep_stage_buffer:
                # If file exists and is empty (no content or just header), delete it
                if os.path.exists(self.sleep_stage_csv_path):
                    with open(self.sleep_stage_csv_path, 'r') as f:
                        content = f.read().strip()
                        # Delete if file is completely empty or contains only the header
                        if content == "" or content == "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id":
                            os.remove(self.sleep_stage_csv_path)
                return True
            
            # Then check if we have a path to save to
            if not self.sleep_stage_csv_path:
                raise MissingOutputPathError(f"Sleep stage buffer is full (size: {self.sleep_stage_buffer_size}) and no output path is set")
            
            # Convert buffer to numpy array
            data_array = np.array(self.sleep_stage_buffer, dtype=float)
            
            # Create format specifiers:
            # - timestamps: match BrainFlow's 6 decimal places for exact matching
            # - sleep stage and buffer ID: use integer format since they're discrete values
            fmt = ['%.6f', '%.6f', '%.0f', '%.0f']
            
            # Save with exact format matching
            if os.path.exists(self.sleep_stage_csv_path):
                # Check if file has content
                with open(self.sleep_stage_csv_path, 'r') as f:
                    content = f.read().strip()
                    has_content = content and content != "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id"
                
                # Append to existing file if it has content
                mode = 'a' if has_content else 'w'
                with open(self.sleep_stage_csv_path, mode) as f:
                    # Write header if this is the first write
                    if not has_content:
                        f.write("timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n")
                    np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
            else:
                # Create new file with header and data
                self._create_sleep_stage_file(data_array)
            
            # Clear buffer
            self.sleep_stage_buffer.clear()
            
            return True
            
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save sleep stage CSV due to I/O error: {e}")
            raise CSVExportError(f"Failed to save sleep stage CSV due to I/O error: {e}")
        except ValueError as e:
            self.logger.error(f"Failed to save sleep stage CSV due to invalid data: {e}")
            raise CSVDataError(f"Failed to save sleep stage CSV due to invalid data: {e}")
        except MissingOutputPathError:
            raise  # Re-raise MissingOutputPathError without wrapping
        except Exception as e:
            self.logger.error(f"Failed to save sleep stage CSV: {e}")
            raise CSVExportError(f"Failed to save sleep stage CSV: {e}")

    def merge_files(self, main_csv_path: Union[str, Path],
                   sleep_stage_csv_path: Union[str, Path],
                   output_path: Union[str, Path]) -> bool:
        """Merge main CSV and sleep stage CSV files into a single output file.
        
        This method matches timestamps between files and creates a final merged output
        with all data properly aligned. The main CSV contains raw BrainFlow data,
        and the sleep stage CSV contains sleep stage classifications and buffer IDs.
        
        # IMPORTANT:
        # Each sleep stage entry is only assigned to the exact end timestamp of its epoch.
        # All other timestamps will have NaN values for sleep stage and buffer ID.
        # While epochs may overlap due to the round-robin buffer system, each epoch's end timestamp is unique,
        # so only one sleep stage value will ever be assigned to a given timestamp.
        #
        # COMMON PITFALL:
        # Do NOT assume that only the end of the current epoch should have a value. If a timestamp is the end of a previous epoch,
        # it should also have a value, even if it overlaps with a non-end timestamp of the current epoch. 
        # A timestamp can be the end of a previous epoch while also being part of the current epoch.
        
        If the sleep stage file doesn't exist, the merge will still proceed, but all
        sleep stage and buffer ID values will be NaN.
        
        Args:
            main_csv_path (Union[str, Path]): Path to main CSV file (raw EEG data)
            sleep_stage_csv_path (Union[str, Path]): Path to sleep stage CSV file
            output_path (Union[str, Path]): Path where to save the merged file
        
        Returns:
            bool: True if merge was successful
        
        Raises:
            CSVExportError: If merge operation fails
            CSVDataError: If data validation fails
            CSVFormatError: If CSV format is incorrect
        """
        try:
            # Validate paths
            main_path = self._validate_file_path(main_csv_path)
            sleep_stage_path = self._validate_file_path(sleep_stage_csv_path)
            output_path = self._validate_file_path(output_path)
            
            # Read main CSV
            # BrainFlow data doesn't have headers, so we need to determine column names from the data
            # First read a single line to get the number of columns
            with open(main_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:  # Check if file is empty
                    raise CSVDataError("Main CSV file is empty")
                num_columns = len(first_line.split('\t'))
            
            # Get timestamp channel index from BrainFlow
            timestamp_channel = self._get_timestamp_channel_index()
            if timestamp_channel >= num_columns:
                raise CSVFormatError(f"Timestamp channel index {timestamp_channel} exceeds number of columns {num_columns}")
            
            # Create column names: channel data with timestamp in correct position
            column_names = [f'channel_{i}' for i in range(num_columns)]
            column_names[timestamp_channel] = 'timestamp'
            
            # Now read the full file with the correct column names
            main_df = pd.read_csv(main_path, delimiter='\t', names=column_names)
            if main_df.empty:
                raise CSVDataError("Main CSV file is empty")
            
            # Store original string timestamps before converting to numeric
            main_df['timestamp_str'] = main_df['timestamp'].astype(str)

            # Create a copy of the main dataframe
            merged_df = main_df.copy()
            
            # Initialize sleep stage and buffer ID columns with NaN
            merged_df['sleep_stage'] = np.nan
            merged_df['buffer_id'] = np.nan
            
            # Check if sleep stage file exists
            if os.path.exists(sleep_stage_path):
                # First validate the file format by checking header and first line
                with open(sleep_stage_path, 'r') as f:
                    header = f.readline().strip()
                    if not header:
                        raise CSVFormatError("Sleep stage CSV file is empty")
                    
                    # Validate header and first line using the new validation function
                    first_line = f.readline().strip()
                    validate_sleep_stage_csv_format(header, first_line if first_line else None)

                # Read sleep stage CSV
                sleep_stage_df = pd.read_csv(sleep_stage_path, delimiter='\t')
                if sleep_stage_df.empty:
                    self.logger.warning("Sleep stage CSV file is empty")
                else:
                    # Store original string timestamps before converting to numeric
                    sleep_stage_df['timestamp_end_str'] = sleep_stage_df['timestamp_end'].astype(str)
                    
                    # Ensure timestamp columns are numeric for range operations
                    try:
                        sleep_stage_df['timestamp_start'] = pd.to_numeric(sleep_stage_df['timestamp_start'], errors='raise')
                        sleep_stage_df['timestamp_end'] = pd.to_numeric(sleep_stage_df['timestamp_end'], errors='raise')
                    except ValueError as e:
                        raise CSVFormatError(f"Invalid timestamp format: {e}")

                    # Sort sleep stage dataframe by timestamp
                    sleep_stage_df = sleep_stage_df.sort_values('timestamp_start')
                    
                    # For each sleep stage entry, find the exact end timestamp and assign values
                    for sleep_row in sleep_stage_df.itertuples():
                        # Find the exact matching timestamp in the main CSV
                        # Note: We use exact string comparison because:
                        # 1. Timestamps are generated with fixed sampling rates (e.g. 125 Hz)
                        # 2. Each sleep stage must be assigned to its exact end timestamp
                        # 3. Using string comparison avoids floating point precision issues
                        # 4. This ensures data integrity by preventing sleep stages from being assigned to nearby timestamps
                        end_mask = merged_df['timestamp_str'] == sleep_row.timestamp_end_str
                        matching_samples = merged_df[end_mask]
                        
                        if matching_samples.empty:
                            self.logger.error(
                                f"No matching timestamp found for sleep stage end timestamp {sleep_row.timestamp_end_str}. "
                                f"This is an error because every sleep stage end timestamp should have a matching sample."
                            )
                            raise CSVDataError(
                                f"No matching timestamp found for sleep stage end timestamp {sleep_row.timestamp_end_str}"
                            )
                            
                        # Check if we would overwrite any non-NaN values
                        if not matching_samples['sleep_stage'].isna().all() or not matching_samples['buffer_id'].isna().all():
                            self.logger.error(
                                f"Attempting to overwrite non-NaN values at timestamp {sleep_row.timestamp_end_str}. "
                                f"Current values - Sleep Stage: {matching_samples['sleep_stage'].iloc[0]}, "
                                f"Buffer ID: {matching_samples['buffer_id'].iloc[0]}"
                            )
                            raise CSVDataError("Cannot overwrite existing sleep stage or buffer ID values")
                            
                        # Assign the sleep stage and buffer ID to the matching samples
                        merged_df.loc[end_mask, 'sleep_stage'] = sleep_row.sleep_stage
                        merged_df.loc[end_mask, 'buffer_id'] = sleep_row.buffer_id
            else:
                self.logger.info(f"Sleep stage file not found at {sleep_stage_path}. Proceeding with merge using NaN values for sleep stage and buffer ID.")
            
            # Ensure main timestamp column is numeric
            try:
                merged_df['timestamp'] = pd.to_numeric(merged_df['timestamp'], errors='raise')
            except ValueError as e:
                raise CSVFormatError(f"Invalid timestamp format in main CSV: {e}")
            
            # Sort main dataframe by timestamp
            merged_df = merged_df.sort_values('timestamp')
            
            # Save merged data
            merged_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
            
            return True
            
        except pd.errors.EmptyDataError:
            raise CSVDataError("One or more CSV files are empty")
        except pd.errors.ParserError as e:
            raise CSVFormatError(f"Failed to parse CSV file: {e}")
        except (CSVDataError, CSVFormatError, CSVExportError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            self.logger.error(f"Failed to merge files: {str(e)}")
            raise CSVExportError(f"Failed to merge files: {e}")

    def _get_timestamp_channel_index(self) -> int:
        """Get the timestamp channel index for the current board.
        
        Returns:
            int: Index of the timestamp channel
            
        Raises:
            CSVExportError: If board_shim is not set
        """
        if self.board_shim is None:
            raise CSVExportError("board_shim is not set; cannot determine timestamp channel index")
        return self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())

    def _handle_buffer_error(self, error: Exception) -> None:
        """Handle buffer-related errors.
        
        Args:
            error (Exception): Error to handle
            
        Raises:
            BufferError: Appropriate buffer error based on the input error
        """
        # TODO: Implement error handling
        pass 