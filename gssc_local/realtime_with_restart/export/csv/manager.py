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

Data Shape Conventions:
    - Input data (new_data): Shape (n_channels, n_samples)
        - Each row represents a channel
        - Each column represents a time point
        - Example: For 8 channels and 1000 samples: shape (8, 1000)
        - This is the raw data format from the board
    
    - Internal buffer (main_csv_buffer): List of lists
        - Each inner list represents one time point
        - Each inner list contains values for all channels at that time point
        - Example: For 8 channels and 1000 samples:
            [
                [sample1_ch1, sample1_ch2, ..., sample1_ch8],  # First time point
                [sample2_ch1, sample2_ch2, ..., sample2_ch8],  # Second time point
                ...
                [sample1000_ch1, sample1000_ch2, ..., sample1000_ch8]  # Last time point
            ]
    
    - CSV file format:
        - Each row represents one time point
        - Each column represents one channel
        - Example: For 8 channels and 1000 samples:
            timestamp    ch1    ch2    ch3    ch4    ch5    ch6    ch7    ch8
            1.234567    0.123  0.456  0.789  ...    ...    ...    ...    ...
            1.234575    0.124  0.457  0.790  ...    ...    ...    ...    ...
            ...

See individual method docstrings for detailed documentation.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union, Dict
import os
from pathlib import Path
import pandas as pd
from .exceptions import (
    CSVExportError, CSVDataError, CSVFormatError,
    MissingOutputPathError, BufferOverflowError,
)
from .utils import (
    get_column_count_from_first_line,
    get_default_sleep_stage_path,
    clean_empty_sleep_stage_file,
    create_format_string,
    create_format_specifiers,
    create_column_names,
    transform_data_to_rows,
    filter_duplicate_timestamps,
    process_sleep_stage_data,
    finalize_merged_data,
    MAIN_DATA_FMT,
    check_file_exists,
)
from .validation import (
    validate_file_path,
    validate_sleep_stage_data,
    validate_timestamps_unique,
    validate_transformed_rows_not_empty,
    validate_add_to_buffer_requirements,
    validate_output_path_set,
    find_duplicates,
    validate_main_csv_columns,
    validate_sleep_stage_timestamps,
    validate_sleep_stage_format,
    validate_board_shim_set,
    validate_csv_not_empty,
    validate_no_sleep_stage_overwrites,
    validate_matching_timestamps
)

class CSVManager:
    """Manages CSV data export and validation for BrainFlow data.
    
    Key Features:
    - Memory-efficient buffer management with incremental saving to prevent overflow
    - Saving BrainFlow data to CSV files with exact format preservation
    - Managing sleep stage data integration with BrainFlow data
    - Testing CSV format against original BrainFlow source (test only) TODO: Move to a test
    
    Data Shape Conventions:
        - Input data (new_data): Shape (n_channels, n_samples)
            - Each row represents a channel
            - Each column represents a time point
            - Example: For 8 channels and 1000 samples: shape (8, 1000)
            - This is the raw data format from the board
        
        - Internal buffer (main_csv_buffer): List of lists
            - Each inner list represents one time point
            - Each inner list contains values for all channels at that time point
            - Example: For 8 channels and 1000 samples:
                [
                    [sample1_ch1, sample1_ch2, ..., sample1_ch8],  # First time point
                    [sample2_ch1, sample2_ch2, ..., sample2_ch8],  # Second time point
                    ...
                    [sample1000_ch1, sample1000_ch2, ..., sample1000_ch8]  # Last time point
                ]
        
        - CSV file format:
            - Each row represents one time point
            - Each column represents one channel
            - Example: For 8 channels and 1000 samples:
                timestamp    ch1    ch2    ch3    ch4    ch5    ch6    ch7    ch8
                1.234567    0.123  0.456  0.789  ...    ...    ...    ...    ...
                1.234575    0.124  0.457  0.790  ...    ...    ...    ...    ...
                ...
    
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
    
    # Format specifiers for sleep stage data:
    # - timestamps: match BrainFlow's 6 decimal places
    # - sleep stage and buffer ID: integer format
    SLEEP_STAGE_FMT = ['%.6f', '%.6f', '%.0f', '%.0f']

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
            rows_to_add, duplicate_count = filter_duplicate_timestamps(new_rows, timestamp_channel, self.last_saved_timestamp, self.logger)
            
            if duplicate_count > 0:
                self.logger.debug(f"Skipped {duplicate_count} duplicate/overlapping samples from streaming")
            
            # Add filtered rows to buffer
            if rows_to_add:
                self.main_csv_buffer.extend(rows_to_add)
            else:
                self.logger.debug("No new samples to add after filtering duplicates")

    def queue_data_for_csv_write(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
        """Queue new data for CSV writing and handle buffer management.

        Handles continuous streaming data that needs careful timestamp management and duplicate prevention.
        
        This method handles:
        - Data validation
        - Managing possible duplicate timestamps from stream
        - Buffer size checks
        - Triggering saves when buffer is full
        - Initial vs subsequent data handling
        - Buffer clearing when necessary
        
        Note on Data Format:
        - Input data must be in (n_channels, n_samples) format (BrainFlow's native format)
        - Data is transformed to (samples, channels) format only at the final CSV writing stage
        - This transformation happens in transform_data_to_rows() and is necessary because:
            1. CSV files are naturally row-oriented (each row represents one time point)
            2. The final CSV format needs to be (samples, channels) for compatibility with other tools
        
        Args:
            new_data (np.ndarray): New data to add in (n_channels, n_samples) format.
                - Each row represents a channel
                - Each column represents a time point
                - Example: For 8 channels and 1000 samples: shape (8, 1000)
                - This is the raw data format from the board
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
            transformed_rows = transform_data_to_rows(new_data, self.logger)

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
            validate_output_path_set(self.main_csv_path, "main CSV")
            validate_output_path_set(self.sleep_stage_csv_path, "sleep stage CSV")

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
            validate_output_path_set(path_to_use, "main CSV", "No main CSV path provided and no existing main_csv_path set")
            
            # Set main path
            self.main_csv_path = str(path_to_use)
            
            # If sleep stage path not set, use convention
            if self.sleep_stage_csv_path is None:
                self.sleep_stage_csv_path = get_default_sleep_stage_path(path_to_use)
            
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
                validate_output_path_set(
                    self.sleep_stage_csv_path,
                    "sleep stage CSV",
                    custom_message=f"Sleep stage buffer is full (size: {self.sleep_stage_buffer_size}) and no output path is set"
                )
                self.save_sleep_stages_to_csv()  # This will handle path validation and errors
            
            # Prepare and add the new entry
            sleep_stage_entry = (float(timestamp_start), float(timestamp_end), float(sleep_stage), float(buffer_id))
            self.sleep_stage_buffer.append(sleep_stage_entry)
            
            return True
            
        except MissingOutputPathError:
            # Let MissingOutputPathError pass through unchanged
            raise
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

    def _prepare_main_csv_data_for_save(self) -> Tuple[np.ndarray, List[str], int]:
        """Prepare main CSV buffer data for saving by validating and converting formats.
        
        Returns:
            Tuple containing:
            - np.ndarray: Prepared data array
            - List[str]: Format specifiers for each column
            - int: Index of timestamp channel
            
        Raises:
            CSVDataError: If data validation fails
        """
        timestamp_channel = self._get_timestamp_channel_index()
        
        # Validate timestamps in buffer
        buffer_timestamps = [row[timestamp_channel] for row in self.main_csv_buffer]
        validate_timestamps_unique(buffer_timestamps, self.logger)
        
        # Convert to numpy array and create format specifiers
        data_array = np.array(self.main_csv_buffer, dtype=float)
        fmt = create_format_specifiers(data_array.shape[1])
        
        return data_array, fmt, timestamp_channel

    def _write_main_csv_data_to_file(self, data_array: np.ndarray, fmt: List[str], is_append: bool = True) -> None:
        """Write data array to main CSV file.
        
        Args:
            data_array: Numpy array of data to write
            fmt: Format specifiers for each column
            is_append: Whether to append to existing file
            
        Raises:
            CSVExportError: If file operations fail
        """
        try:
            mode = 'a' if is_append else 'w'
            with open(self.main_csv_path, mode) as f:
                np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
        except (IOError, OSError) as e:
            raise CSVExportError(f"Failed to {'append to' if is_append else 'create'} main CSV file: {e}")



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
            # Validate requirements
            validate_output_path_set(self.main_csv_path, "main CSV")
            if not self.main_csv_buffer:
                return True

            # Prepare data for saving
            data_array, fmt, timestamp_channel = self._prepare_main_csv_data_for_save()
            
            # Determine if we're appending or creating new file
            file_exists = os.path.exists(self.main_csv_path)
            
            # Write data to file
            self._write_main_csv_data_to_file(data_array, fmt, is_append=file_exists)
            
            # Update state
            self._update_last_saved_timestamp(timestamp_channel)
            self.main_csv_buffer.clear()
            
            return True
            
        except (CSVExportError, CSVDataError):
            # Re-raise known exceptions
            raise
        except ValueError as e:
            self.logger.error(f"Failed to save main CSV due to invalid data: {e}")
            raise CSVDataError(f"Failed to save main CSV due to invalid data: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while saving main CSV: {e}")
            raise CSVExportError(f"Unexpected error while saving main CSV: {e}")

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
                    np.savetxt(f, data_array, delimiter='\t', fmt=self.SLEEP_STAGE_FMT)

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to create sleep stage file: {e}")
            raise CSVExportError(f"Failed to create sleep stage file: {e}")



    def _prepare_sleep_stage_data(self) -> Tuple[np.ndarray, List[str]]:
        """Convert sleep stage buffer to numpy array and prepare format specifiers.
        
        Returns:
            Tuple containing:
            - np.ndarray: Sleep stage data array
            - List[str]: Format specifiers for each column
        """
        data_array = np.array(self.sleep_stage_buffer, dtype=float)
        return data_array, self.SLEEP_STAGE_FMT

    def _append_to_sleep_stage_file(self, data_array: np.ndarray, fmt: List[str]) -> None:
        """Append data to existing sleep stage file, handling header if needed.
        
        Args:
            data_array: Numpy array of sleep stage data
            fmt: Format specifiers for each column
        """
        with open(self.sleep_stage_csv_path, 'r') as f:
            content = f.read().strip()
            has_content = content and content != "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id"
        
        mode = 'a' if has_content else 'w'
        with open(self.sleep_stage_csv_path, mode) as f:
            if not has_content:
                f.write("timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n")
            np.savetxt(f, data_array, delimiter='\t', fmt=fmt)

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
            # Handle empty buffer case
            if not self.sleep_stage_buffer:
                clean_empty_sleep_stage_file(self.sleep_stage_csv_path)
                return True
            
            # Validate path exists using existing validation function
            validate_output_path_set(self.sleep_stage_csv_path, "sleep stage CSV")
            
            # Prepare data
            data_array, fmt = self._prepare_sleep_stage_data()
            
            # Write data
            if os.path.exists(self.sleep_stage_csv_path):
                self._append_to_sleep_stage_file(data_array, fmt)
            else:
                self._create_sleep_stage_file(data_array)
            
            # Clear buffer after successful write
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


    
    def _read_and_validate_main_csv(self, main_path: Path) -> pd.DataFrame:
        """Read and validate the main CSV file containing BrainFlow data.
        
        Args:
            main_path (Path): Path to the main CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing the main CSV data with proper column names
            
        Raises:
            CSVDataError: If file is empty
            CSVFormatError: If format is invalid
        """
        # Validate file path
        main_path = validate_file_path(main_path)
        
        # Get column count from first line
        num_columns = get_column_count_from_first_line(main_path, self.logger)
        
        # Get and validate timestamp channel
        timestamp_channel = self._get_timestamp_channel_index()
        validate_main_csv_columns(num_columns, timestamp_channel)
        
        # Create column names and read full file
        column_names = create_column_names(num_columns, timestamp_channel)
        main_df = pd.read_csv(main_path, delimiter='\t', names=column_names)
        
        # Add string timestamp column
        main_df['timestamp_str'] = main_df['timestamp'].astype(str)
        return main_df

    def _read_and_validate_sleep_stage_csv(self, sleep_stage_path: Path) -> Optional[pd.DataFrame]:
        """Read and validate the sleep stage CSV file.
        
        Args:
            sleep_stage_path (Path): Path to the sleep stage CSV file
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing sleep stage data, or None if file doesn't exist
            
        Raises:
            CSVFormatError: If format is invalid
        """
        # Check file existence
        if not check_file_exists(sleep_stage_path, self.logger):
            return None
            
        # Validate CSV format
        validate_sleep_stage_format(sleep_stage_path)
        
        # Read and process data
        return process_sleep_stage_data(sleep_stage_path)

    def _merge_sleep_stages_into_main_data(self, merged_df: pd.DataFrame, sleep_stage_df: pd.DataFrame) -> pd.DataFrame:
        """Merge sleep stage data into the main DataFrame.
        
        Args:
            merged_df (pd.DataFrame): Main DataFrame to merge into
            sleep_stage_df (pd.DataFrame): Sleep stage DataFrame to merge from
            
        Returns:
            pd.DataFrame: Merged DataFrame with sleep stages
            
        Raises:
            CSVDataError: If merge validation fails
        """
        # Initialize sleep stage columns
        merged_df['sleep_stage'] = np.nan
        merged_df['buffer_id'] = np.nan
        
        # Process each sleep stage entry
        for sleep_row in sleep_stage_df.itertuples():
            end_mask = merged_df['timestamp_str'] == sleep_row.timestamp_end_str
            matching_samples = merged_df[end_mask]
            
            # Validate timestamps match
            validate_matching_timestamps(matching_samples, sleep_row.timestamp_end_str, self.logger)
            
            # Validate no overwrites
            validate_no_sleep_stage_overwrites(matching_samples, sleep_row.timestamp_end_str, self.logger)
            
            # Assign values
            merged_df.loc[end_mask, 'sleep_stage'] = sleep_row.sleep_stage
            merged_df.loc[end_mask, 'buffer_id'] = sleep_row.buffer_id
            
        return merged_df

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
            main_path = validate_file_path(main_csv_path)
            sleep_stage_path = validate_file_path(sleep_stage_csv_path)
            output_path = validate_file_path(output_path)
            
            # Read and validate input files
            main_df = self._read_and_validate_main_csv(main_path)
            sleep_stage_df = self._read_and_validate_sleep_stage_csv(sleep_stage_path)
            
            # Create initial merged DataFrame
            merged_df = main_df.copy()
            
            # Merge sleep stages if available
            if sleep_stage_df is not None:
                merged_df = self._merge_sleep_stages_into_main_data(merged_df, sleep_stage_df)
            else:
                # Initialize empty sleep stage columns
                merged_df['sleep_stage'] = np.nan
                merged_df['buffer_id'] = np.nan
            
            # Finalize and save
            merged_df = finalize_merged_data(merged_df)
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
        validate_board_shim_set(self.board_shim, self.logger)
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