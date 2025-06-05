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

BREAKING CHANGES:
The following methods have been updated with new buffer management logic:
# TODO: We should eventually remove the old methods and only use the new ones.
- save_new_data_to_csv_buffer() -> add_data_to_buffer()
- add_sleep_stage_to_csv_buffer() -> add_sleep_stage_to_sleep_stage_csv()
- save_to_csv() -> save_all_and_cleanup()


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
    validate_sleep_stage_csv_format
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
        self._validate_data_shape = lambda data: validate_data_shape(data)
        self._validate_file_path = lambda path: validate_file_path(path)
        self.validate_saved_csv_matches_original_source = lambda original_csv_path, output_path=None: validate_saved_csv_matches_original_source(self, original_csv_path, output_path)
    
   
    def _check_main_buffer_overflow(self) -> bool:
        """Check if main buffer (BrainFlow data) would overflow with current size.
        
        This method specifically checks the main_csv_buffer which holds BrainFlow data.
        For sleep stage buffer overflow checks, use _check_sleep_stage_buffer_overflow().
        
        Returns:
            bool: True if main buffer would overflow, False otherwise
        """
        return len(self.main_csv_buffer) >= self.main_buffer_size

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

    def add_data_to_buffer(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
        """Add new data to the buffer and handle buffer management.
        
        This method is the new implementation of save_new_data_to_csv_buffer with
        improved buffer management. It handles:
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
            # Add start logging
            self.logger.info("\n=== CSVManager.add_data_to_buffer [L317] ===")
            self.logger.info(f"Received data shape: {new_data.shape}")
            self.logger.info(f"Current buffer size: {len(self.main_csv_buffer)}")
            self.logger.info(f"Buffer size limit: {self.main_buffer_size}")
            self.logger.info(f"Is initial data: {is_initial}")

            # Log the current CSV file and total samples
            if self.main_csv_path and os.path.exists(self.main_csv_path):
                with open(self.main_csv_path, 'r') as f:
                    total_samples = sum(1 for _ in f)
                self.logger.info(f"Current CSV file: {self.main_csv_path}")
                self.logger.info(f"Total samples in CSV: {total_samples}")
            else:
                self.logger.info("No existing CSV file")

            self._validate_data_shape(new_data)
            
            # Validate that we have an output path if data would exceed buffer size
            if len(new_data.T) > self.main_buffer_size and not self.main_csv_path:
                self.logger.error(f"Missing output path: Initial data size {len(new_data.T)} exceeds buffer size limit {self.main_buffer_size}")
                raise MissingOutputPathError(f"Output path must be set before accepting data that exceeds buffer size limit {self.main_buffer_size}")
            
            # Convert data to list of rows
            new_rows = new_data.T.tolist()

            # Get timestamp channel index
            if self.board_shim is not None:
                timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
            else:
                raise CSVExportError("board_shim is not set; cannot determine timestamp channel index")

            if is_initial:
                self.logger.warning("[DEBUG] add_data_to_buffer: is_initial=True. Treating this as initial data chunk.")
                # Clear the output file early for initial data
                self.clear_output_file()
                # Add all rows for initial data first
                self.main_csv_buffer.extend(new_rows)
                if new_rows:
                    self.last_saved_timestamp = new_rows[-1][timestamp_channel]
                
                # Then check if buffer size exceeds limit
                if len(self.main_csv_buffer) > self.main_buffer_size:
                    # if the data is too large for the buffer, save data to the csv
                    self.logger.info(f"Initial data size {len(new_rows)} exceeds buffer size limit {self.main_buffer_size}. This is expected behavior. Saving data to csv.")
                    # Store the length before saving
                    data_size = len(new_rows)
                    self.logger.info(f"[L392] Buffer before save_incremental_to_csv: size={len(self.main_csv_buffer)}")
                    self.save_incremental_to_csv(is_initial=True)  # This will save and clear the buffer
                    self.logger.info(f"[L394] Buffer after save_incremental_to_csv: size={len(self.main_csv_buffer)}")
                    self.last_saved_timestamp = None
            else:
                # For subsequent data, filter out exact duplicates
                if self.last_saved_timestamp is not None:
                    # Find the first row with a timestamp greater than the last saved timestamp
                    start_idx = 0
                    for i, row in enumerate(new_rows):
                        if row[timestamp_channel] > self.last_saved_timestamp:
                            start_idx = i
                            break
                    
                    rows_to_add = new_rows[start_idx:]
                    
                    # Add rows first
                    self.main_csv_buffer.extend(rows_to_add)
                    if rows_to_add:
                        self.last_saved_timestamp = rows_to_add[-1][timestamp_channel]
                    
                    # Then check if buffer size exceeds limit
                    if len(self.main_csv_buffer) > self.main_buffer_size:
                        self.logger.info(f"Saving current buffer due to buffer overflow. This is expected behavior. Current size: {len(self.main_csv_buffer)}, Adding: {len(rows_to_add)}, Limit: {self.main_buffer_size}.")
                        # Save current buffer
                        self.logger.info(f"[L378] Buffer before save_incremental_to_csv: size={len(self.main_csv_buffer)}")
                        self.save_incremental_to_csv()  # This clears the buffer
                        self.logger.info(f"[L380] Buffer after save_incremental_to_csv: size={len(self.main_csv_buffer)}")
                else: # if no last saved timestamp
                    # TODO: This seems like a duplicate of the code above in the block for if is_initial:
                    # TODO: continued... Isn't "If no last saved timestamp" the same as if is_initial?
                    # If no last saved timestamp, add all rows first
                    self.main_csv_buffer.extend(new_rows)
                    if new_rows:
                        self.last_saved_timestamp = new_rows[-1][timestamp_channel]
                    
                    # Then check if buffer size exceeds limit
                    if len(self.main_csv_buffer) > self.main_buffer_size:
                        self.logger.info(f"Saving current buffer due to buffer overflow. This is expected behavior. Current size: {len(self.main_csv_buffer)}, Adding: {len(new_rows)}, Limit: {self.main_buffer_size}")
                        # Save current buffer
                        self.logger.info(f"[L399] Buffer before save_incremental_to_csv: size={len(self.main_csv_buffer)}")
                        self.save_incremental_to_csv()  # This clears the buffer
                        self.logger.info(f"[L401] Buffer after save_incremental_to_csv: size={len(self.main_csv_buffer)}")
            
            # Add end logging
            self.logger.info(f"Buffer size after adding data: {len(self.main_csv_buffer)}")
            if self.main_csv_path and os.path.exists(self.main_csv_path):
                with open(self.main_csv_path, 'r') as f:
                    total_samples = sum(1 for _ in f)
                self.logger.info(f"Total samples in CSV after save: {total_samples}")
            self.logger.info("=== End add_data_to_buffer ===\n")

            return True
            
        except CSVDataError as e:
            self.logger.error(f"Failed to add data to buffer: {e}")
            raise
        except BufferOverflowError as e:
            self.logger.error(f"Buffer overflow: {e}")
            raise

    def save_new_data_to_csv_buffer(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
        """Legacy method that now uses add_data_to_buffer.
        TODO: This method is deprecated and should be deleted. Use add_data_to_buffer() instead.

        BREAKING CHANGE: This method's behavior has changed to use the new buffer management system.
        It no longer behaves exactly as it did in previous versions.
        
        This method is kept for backward compatibility and will be deprecated in a future version.
        Please use add_data_to_buffer() instead.
        
        Args:
            new_data (np.ndarray): New data to save (channels x samples)
            is_initial (bool): Whether this is the initial data chunk
            
        Returns:
            bool: True if data was saved successfully
            
        Raises:
            CSVDataError: If data validation fails
            BufferOverflowError: If adding data would exceed buffer size limit
        """
        import warnings
        warnings.warn("save_new_data_to_csv_buffer() is deprecated. Please use add_data_to_buffer() instead.", DeprecationWarning)
        return self.add_data_to_buffer(new_data, is_initial)

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
            self.logger.info("\n=== CSVManager.save_all_data() ===")
            self.logger.info(f"Main buffer size: {len(self.main_csv_buffer)}")
            self.logger.info(f"Sleep stage buffer size: {len(self.sleep_stage_buffer)}")
            
            # Debug logging
            print(f"\n=== Debug: CSVManager.save_all_data ===")
            print(f"Main buffer size: {self.main_buffer_size}")
            print(f"Main buffer length: {len(self.main_csv_buffer)}")
            print(f"Sleep stage buffer length: {len(self.sleep_stage_buffer)}")
            
            # If no data to save, just return success
            if not self.main_csv_buffer and not self.sleep_stage_buffer:
                self.logger.info("No data in buffers to save")
                return True
            
            # Only check paths if we have data to save
            if not self.main_csv_path:
                raise CSVExportError("Main CSV path not configured")
            if not self.sleep_stage_csv_path:
                raise CSVExportError("Sleep stage CSV path not configured")
            
            self.logger.info(f"Main CSV path: {self.main_csv_path}")
            self.logger.info(f"Sleep stage CSV path: {self.sleep_stage_csv_path}")
            
            # Save any remaining data in the main buffer
            if self.main_csv_buffer:
                self.logger.info(f"Saving {len(self.main_csv_buffer)} remaining rows")
                self.save_incremental_to_csv()
                
            # Save any remaining sleep stage data
            if self.sleep_stage_buffer:
                self.logger.info(f"Saving {len(self.sleep_stage_buffer)} remaining sleep stage entries")
                self.save_sleep_stages_to_csv()
                
            self.logger.info("=== End save_all_data() ===\n")
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
                self.logger.info(f"Using default sleep stage path: {self.sleep_stage_csv_path}")
            
            result = self.save_all_data()

            if merge_files:
                self.merge_files(self.main_csv_path, self.sleep_stage_csv_path, merge_output_path)

            self.cleanup(reset_paths=True)
            self.logger.info("=== End save_all_and_cleanup() ===\n")
            return result
        except Exception as e:
            self.logger.error(f"Failed to save all data and cleanup: {e}")
            raise CSVExportError(f"Failed to save all data and cleanup: {e}")

    def save_to_csv(self) -> bool:
        """Legacy method that now uses save_all_data and cleanup.
        TODO: This method is deprecated and should be deleted. Use save_all_and_cleanup() instead.
        
        BREAKING CHANGE: This method's behavior has changed to use the new buffer management system.
        It no longer behaves exactly as it did in previous versions.
        
        This method is kept for backward compatibility and will be deprecated in a future version.
        Please use save_all_data() followed by cleanup() instead.
        
        Returns:
            bool: True if save was successful
            
        Raises:
            CSVExportError: If save operation fails
            CSVDataError: If data validation fails
        """
        import warnings
        self.logger.warning("save_to_csv() is deprecated. Please use save_all_data() followed by cleanup() instead.")
        warnings.warn("save_to_csv() is deprecated. Please use save_all_data() followed by cleanup() instead.", DeprecationWarning)
        result = self.save_all_data()
        self.cleanup(reset_paths=True)
        return result
    

    
    def add_sleep_stage_to_sleep_stage_csv(self, sleep_stage: float, buffer_id: float, timestamp_start: float, timestamp_end: float) -> bool:
        """Add sleep stage data to the sleep stage buffer.
        
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
            MissingOutputPathError: If adding data would exceed buffer size limit
        """
        try:
            self.logger.info(f"\n=== Adding sleep stage ===")
            self.logger.info(f"Sleep stage: {sleep_stage}")
            self.logger.info(f"Buffer ID: {buffer_id}")
            self.logger.info(f"Timestamp range: {timestamp_start:.2f} to {timestamp_end:.2f}")
            self.logger.info(f"Current buffer size: {len(self.sleep_stage_buffer)}")
            
            # Validate inputs using the new validation function
            validate_sleep_stage_data(sleep_stage, buffer_id, timestamp_start, timestamp_end)
            
            # Check if adding this entry would exceed buffer size
            if len(self.sleep_stage_buffer) >= self.sleep_stage_buffer_size:
                if self.sleep_stage_csv_path:
                    # Save current buffer contents
                    entries_to_save = self.sleep_stage_buffer.copy()
                    if entries_to_save:
                        data_array = np.array(entries_to_save, dtype=float)
                        fmt = ['%.6f', '%.6f', '%.0f', '%.0f']
                        # Check if file exists and has content
                        file_exists = os.path.exists(self.sleep_stage_csv_path)
                        has_content = False
                        if file_exists:
                            with open(self.sleep_stage_csv_path, 'r') as f:
                                content = f.read().strip()
                                has_content = content and content != "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id"
                        
                        # Open file in appropriate mode
                        mode = 'a' if has_content else 'w'
                        with open(self.sleep_stage_csv_path, mode) as f:
                            # Write header if this is the first write
                            if not has_content:
                                f.write("timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n")
                            np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
                        self.logger.debug(f"Saved {len(entries_to_save)} entries to {self.sleep_stage_csv_path}")
                    # Clear buffer
                    self.sleep_stage_buffer.clear()
                    self.logger.debug("Buffer cleared")
                else:
                    raise MissingOutputPathError(f"Sleep stage buffer is full (size: {self.sleep_stage_buffer_size}) and no output path is set")
            
            # Prepare and add the new entry
            sleep_stage_entry = (float(timestamp_start), float(timestamp_end), float(sleep_stage), float(buffer_id))
            self.sleep_stage_buffer.append(sleep_stage_entry)
            self.logger.debug(f"Added sleep stage entry: {sleep_stage_entry}")
            self.logger.debug(f"Current buffer size: {len(self.sleep_stage_buffer)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add sleep stage to buffer: {e}")
            if isinstance(e, MissingOutputPathError):
                raise
            raise CSVDataError(f"Failed to add sleep stage to buffer: {e}")

    def add_sleep_stage_to_csv_buffer(self, sleep_stage: float, next_buffer_id: float, epoch_end_idx: int) -> bool:
        """Legacy method that is no longer supported due to breaking changes.
        TODO: This method is deprecated and should be deleted. Use add_sleep_stage_to_sleep_stage_csv() instead.
        
        BREAKING CHANGE: This method signature is incompatible with the new buffer management system.
        The new method requires explicit timestamps instead of epoch indices.
        
        Please use add_sleep_stage_to_sleep_stage_csv(sleep_stage, buffer_id, timestamp_start, timestamp_end) instead.
        
        Args:
            sleep_stage (float): Sleep stage classification
            next_buffer_id (float): ID of the next buffer
            epoch_end_idx (int): Index where to add the data
            
        Returns:
            bool: Always raises NotImplementedError
            
        Raises:
            NotImplementedError: This method is no longer supported
        """
        self.logger.warning(f"Deprecated method add_sleep_stage_to_csv_buffer called with sleep_stage={sleep_stage}, next_buffer_id={next_buffer_id}, epoch_end_idx={epoch_end_idx}")
        raise NotImplementedError(
            "add_sleep_stage_to_csv_buffer() is no longer supported due to breaking changes. "
            "Use add_sleep_stage_to_sleep_stage_csv(sleep_stage, buffer_id, timestamp_start, timestamp_end) instead."
        )
    
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
            self.logger.info("Cleaning up CSVManager resources")
            
            # Clear data buffers
            self.logger.info(f"[L747] Buffer before clear: size={len(self.main_csv_buffer)}")
            self.main_csv_buffer.clear()
            self.sleep_stage_buffer.clear()
            
            # Reset state
            self.last_saved_timestamp = None
            
            # Optionally reset paths
            if reset_paths:
                self.main_csv_path = None
                self.sleep_stage_csv_path = None
            
            self.logger.info("CSVManager cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during CSVManager cleanup: {e}")
            raise CSVExportError(f"Failed to cleanup CSVManager: {e}")

    def save_incremental_to_csv(self, is_initial: bool = False) -> bool:
        """Save current buffer contents to CSV file and clear the buffer.
        
        This method handles both first write (create new file) and subsequent writes (append).
        It preserves the exact CSV format and clears the buffer after successful write.
        
        Returns:
            bool: True if save was successful or if there was no data to save
            
        Raises:
            CSVExportError: If save operation fails
            CSVDataError: If data validation fails
        """
        try:
            self.logger.info("\n=== CSVManager.save_incremental_to_csv [L853] ===")
            self.logger.info(f"[L854] Buffer size at start of save_incremental_to_csv: {len(self.main_csv_buffer)}")
            self.logger.info(f"Is initial save: {is_initial}")
            self.logger.info(f"CSV path: {self.main_csv_path}")
            

                
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
                self.logger.debug("Buffer is empty, returning without file operations because save_incremental_to_csv cannot save empty buffers to the CSV file.")
                return True
            
            # Get timestamp channel index
            if self.board_shim is not None:
                timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
            else:
                raise CSVExportError("board_shim is not set; cannot determine timestamp channel index")

            # Validate timestamps in buffer before saving
            buffer_timestamps = [row[timestamp_channel] for row in self.main_csv_buffer]
            unique_timestamps = set(buffer_timestamps)
            if len(buffer_timestamps) != len(unique_timestamps):
                duplicate_count = len(buffer_timestamps) - len(unique_timestamps)
                self.logger.error(f"Found {duplicate_count} duplicate timestamps in buffer before saving!")
                # Find and log the duplicates
                from collections import Counter
                timestamp_counts = Counter(buffer_timestamps)
                duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}
                for ts, count in duplicates.items():
                    self.logger.error(f"Timestamp {ts} appears {count} times")
                raise CSVDataError(f"Found {duplicate_count} duplicate timestamps in buffer before saving")
            
            # Convert to numpy array
            data_array = np.array(self.main_csv_buffer, dtype=float)
            self.logger.info(f"[L888] Data array shape before format specifiers: {data_array.shape}")
            
            # Create format specifiers - all columns use float format
            fmt = ['%.6f'] * data_array.shape[1]
            
            # Save with exact format matching
            if os.path.exists(self.main_csv_path):
                # Log before appending
                with open(self.main_csv_path, 'r') as f:
                    samples_before = sum(1 for _ in f)
                self.logger.info(f"Appending to file. Samples before append: {samples_before}")
                
                # Append to existing file
                with open(self.main_csv_path, 'a') as f:
                    np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
                
                # Log after appending
                with open(self.main_csv_path, 'r') as f:
                    samples_after = sum(1 for _ in f)
                self.logger.info(f"Append complete. Samples after append: {samples_after}")
            else:
                # Log before creating new file
                self.logger.info("Creating new file") # TODO: this is too coupled. WE should be creating the file in the add_data_to_buffer method on the initial data chunk.
                
                # Create a new file
                np.savetxt(self.main_csv_path, data_array, delimiter='\t', fmt=fmt)
                
                # Log after creating new file
                with open(self.main_csv_path, 'r') as f:
                    samples_after = sum(1 for _ in f)
                self.logger.info(f"New file created. Samples in new file: {samples_after}")
            
            # Clear buffer after saving
            if self.main_csv_buffer:
                self.last_saved_timestamp = self.main_csv_buffer[-1][timestamp_channel]
            self.logger.info(f"[L948] Buffer before clear: size={len(self.main_csv_buffer)}")
            self.main_csv_buffer.clear()
            self.logger.info(f"L951 Buffer size after clearing: {len(self.main_csv_buffer)}")
            self.logger.info("=== End save_incremental_to_csv ===\n")
            
            # Log file state after save
            if os.path.exists(self.main_csv_path):
                with open(self.main_csv_path, 'r') as f:
                    total_samples = sum(1 for _ in f)
                self.logger.info(f"Total samples in CSV after save: {total_samples}")
            
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
        """
        try:
            # First check if we have any data to save
            if not self.sleep_stage_buffer:
                self.logger.debug("Sleep stage buffer is empty")
                # If file exists and is empty (no content or just header), delete it
                if os.path.exists(self.sleep_stage_csv_path):
                    self.logger.debug(f"File exists at {self.sleep_stage_csv_path}")
                    with open(self.sleep_stage_csv_path, 'r') as f:
                        content = f.read().strip()
                        self.logger.debug(f"File content: '{content}'")
                        # Delete if file is completely empty or contains only the header
                        if content == "" or content == "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id":
                            self.logger.debug("File is empty or contains only header, deleting")
                            os.remove(self.sleep_stage_csv_path)
                            self.logger.debug("File deleted")
                        else:
                            self.logger.debug("File contains actual data, keeping it")
                return True
            
            # Then check if we have a path to save to
            if not self.sleep_stage_csv_path:
                raise CSVExportError("No sleep stage CSV path set")
            
            # Convert buffer to numpy array
            self.logger.debug(f"Sleep stage buffer before conversion: {self.sleep_stage_buffer}")
            data_array = np.array(self.sleep_stage_buffer, dtype=float)
            self.logger.debug(f"Data array shape: {data_array.shape}")
            self.logger.debug(f"Data array content: {data_array}")
            
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
                header = "timestamp_start\ttimestamp_end\tsleep_stage\tbuffer_id\n"
                with open(self.sleep_stage_csv_path, 'w') as f:
                    f.write(header)
                    np.savetxt(f, data_array, delimiter='\t', fmt=fmt)
            
            # Clear buffer
            self.sleep_stage_buffer.clear()
            
            return True
            
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save sleep stage CSV due to I/O error: {e}")
            raise CSVExportError(f"Failed to save sleep stage CSV due to I/O error: {e}")
        except ValueError as e:
            self.logger.error(f"Failed to save sleep stage CSV due to invalid data: {e}")
            raise CSVDataError(f"Failed to save sleep stage CSV due to invalid data: {e}")
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
            timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
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
            
            logging.info(f"Read main CSV file with {len(main_df)} rows")
            logging.info(f"Main CSV columns: {main_df.columns.tolist()}")
            
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
                    
                    logging.info(f"Read sleep stage CSV file with {len(sleep_stage_df)} rows")
                    logging.info(f"Sleep stage data:\n{sleep_stage_df}")
                    
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
                        self.logger.debug(f"\nDEBUG: Comparing timestamps:")
                        self.logger.debug(f"Sleep stage end timestamp (str): {sleep_row.timestamp_end_str}")
                        self.logger.debug(f"Available main timestamps (first 5): {merged_df['timestamp_str'].head().tolist()}")
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
                        self.logger.debug(f"Assigned sleep stage {sleep_row.sleep_stage} to timestamp {sleep_row.timestamp_end_str}")
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
            logging.info(f"Saved merged file to {output_path}")
            
            return True
            
        except pd.errors.EmptyDataError:
            raise CSVDataError("One or more CSV files are empty")
        except pd.errors.ParserError as e:
            raise CSVFormatError(f"Failed to parse CSV file: {e}")
        except (CSVDataError, CSVFormatError, CSVExportError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            logging.error(f"Failed to merge files: {str(e)}")
            raise CSVExportError(f"Failed to merge files: {e}")





    def _check_sleep_stage_buffer_overflow(self) -> bool:
        """Check if sleep stage buffer would overflow with current size.
        
        Returns:
            bool: True if buffer would overflow, False otherwise
        """
        return len(self.sleep_stage_buffer) >= self.sleep_stage_buffer_size

    def _handle_buffer_error(self, error: Exception) -> None:
        """Handle buffer-related errors.
        
        Args:
            error (Exception): Error to handle
            
        Raises:
            BufferError: Appropriate buffer error based on the input error
        """
        # TODO: Implement error handling
        pass 