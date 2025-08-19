"""
Data Manager for handling data export and validation of brainflow data.

This module provides functionality for saving and validating data from brainflow streaming to a csv file with sleep stage integration.
It implements a memory-efficient buffer management system that prevents memory overflow during long recordings.

Data Shape Conventions:
    - Input data (new_data): Shape (n_channels, n_samples)
        - Each row represents a channel
        - Each column represents a time point
        - Example: For 8 channels and 1000 samples: shape (8, 1000)
        - This is the raw data format from the board
    
    - Internal processing: All internal processing maintains (n_channels, n_samples) format
        - ETDBufferManager maintains data in this format
        - SignalProcessor expects and returns data in this format
        - Only transformed to (samples, channels) at final CSV writing stage
"""

import numpy as np
import logging
import torch
from pathlib import Path
from typing import Union, Optional, Tuple, List
from gssc_local.realtime_with_restart.processor import SignalProcessor
from gssc_local.pyqt_visualizer import PyQtVisualizer
from gssc_local.montage import Montage
from gssc_local.realtime_with_restart.export.csv.manager import CSVManager
from gssc_local.realtime_with_restart.export.csv.test_utils import compare_csv_files
from gssc_local.realtime_with_restart.export.csv.validation import validate_file_path
from gssc_local.realtime_with_restart.export.csv.exceptions import CSVExportError, CSVDataError
from gssc_local.realtime_with_restart.utils.session_utils import generate_session_timestamp
from .etd_buffer_manager import ETDBufferManager
from .channel_mapping import ChannelIndexMapping, NumPyDataWithBrainFlowDataKey, RawBoardDataWithKeys
import pandas as pd
import time
import os
from .core.gap_handler import GapHandler

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data buffers and their processing"""
    def __init__(self, board_shim, sampling_rate, montage: Montage = None, event_dispatcher=None, session_timestamp=None):
        self.board_shim = board_shim
        self.sampling_rate = sampling_rate
        self.montage = montage
        self.event_dispatcher = event_dispatcher 
        self.seconds_per_epoch = 30
        self.seconds_per_step = 5
        # Buffer configuration
        self.points_per_epoch = self.seconds_per_epoch * sampling_rate  # 30 second epochs
        self.points_per_step = self.seconds_per_step * sampling_rate    # 5 second steps
        self.buffer_start = 0
        self.buffer_end = self.seconds_per_epoch
        self.round_robin_delay = self.seconds_per_step
        
        # Initialize channels and buffers
        self.electrode_channels, self.board_timestamp_channel, electrode_and_timestamp_channels = self._init_channels()
        
        # Initialize buffer manager
        self.etd_buffer_manager = ETDBufferManager(
            max_buffer_size=35 * sampling_rate,  # 35 seconds of data
            timestamp_board_key=self.board_timestamp_channel,
            channel_count=len(electrode_and_timestamp_channels),
            electrode_and_timestamp_board_keys=electrode_and_timestamp_channels
        )
        
        # Buffer tracking
        self.last_processed_buffer = -1
        
        # Optimized tracking of processed epochs - only store last epoch per buffer instead of full history
        # Each element is either None (no epochs processed) or (start_idx_abs, end_idx_abs) tuple for last epoch
        # Memory efficient: O(1) instead of O(total_epochs_processed)
        self.last_processed_epoch_per_buffer = [None] * 6  # Last processed epoch per buffer
        self.epochs_processed_count_per_buffer = [0] * 6   # Count of epochs processed per buffer (for monitoring)
        
        # Initialize hidden states for each buffer
        self.buffer_hidden_states = [
            [torch.zeros(10, 1, 256) for _ in range(7)]  # 7 hidden states for 7 combinations
            for _ in range(6)  # 6 buffers (0s to 25s in 5s steps)
        ]
        self.signal_processor = SignalProcessor(use_cuda=False)
        # self.visualizer = Visualizer(self.seconds_per_epoch, self.board_shim, montage)
        self.visualizer = PyQtVisualizer(self.seconds_per_epoch, self.board_shim, montage)
        self.expected_interval = 1.0 / sampling_rate
        self.timestamp_tolerance = self.expected_interval * 0.01  # 1% tolerance
        self.gap_threshold = 2.0  # Large gap threshold (seconds)
        self.interpolation_threshold = 0.1  # Maximum gap to interpolate (seconds)
        self.current_epoch_start_time = None
        # Index position of timestamp within electrode_and_timestamp_data array
        
        # Add validation settings
        self.validate_consecutive_values = False  # Set to True to enable consecutive value validation
        self.validation_channel = 0  # Channel to validate (default to first channel)
        self.last_validated_value_for_consecutive_data_validation = None  # Track last validated value
        self.output_csv_path = None
        self.last_saved_timestamp = None  # Track last saved timestamp to prevent duplicates
        
        # Generate timestamp for session if not provided
        if session_timestamp is None:
            session_timestamp = generate_session_timestamp()
        self.session_timestamp = session_timestamp
        
        # Create timestamped CSV filenames
        main_csv_filename = f'data_{session_timestamp}.csv'
        sleep_stage_csv_filename = f'sleep_stages_{session_timestamp}.csv'
        
        # Initialize CSVManager
        self.csv_manager = CSVManager(
            self.board_shim,
            main_buffer_size=1000,  # Increased from 500
            sleep_stage_buffer_size=200,  # Kept the same
            main_csv_path=main_csv_filename,
            sleep_stage_csv_path=sleep_stage_csv_filename
        )
        
        # Initialize gap handler
        self.gap_handler = GapHandler(sampling_rate=sampling_rate, gap_threshold=2.0)
        
        # Add epoch scoring counter
        self.epochs_scored = 0
    


    def _init_channels(self):
        """Initialize and validate channel information from the board.
        
        Returns:
            tuple: Contains:
                - electrode_channels (list): Channel numbers for EEG/EMG/EOG on the board
                - board_timestamp_channel (int): Channel number for timestamp on the board
                - electrode_and_timestamp_channels (list): Combined list of all channel numbers we'll read
                
        Raises:
            RuntimeError: If board is not properly initialized
        """
        if not self.board_shim:
            raise RuntimeError("Board not initialized properly")
            
        try:
            # Get channel information from board
            electrode_channels = self._get_board_electrode_channels()
            board_timestamp_channel = self._get_board_timestamp_channel()
            
            # Combine channels into unified list
            electrode_and_timestamp_channels = self._combine_channels(electrode_channels, board_timestamp_channel)
            
            return electrode_channels, board_timestamp_channel, electrode_and_timestamp_channels
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize channels: {str(e)}")
            
    def _get_board_electrode_channels(self):
        """Get the electrode (EEG/EMG/EOG) channel indices from the board."""
        return self.board_shim.get_exg_channels(self.board_shim.get_board_id())
        
    def _get_board_timestamp_channel(self):
        """Get the timestamp channel index from the board."""
        return self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
        
    def _combine_channels(self, electrode_channels, timestamp_channel):
        """Combine electrode and timestamp channels into a unified list.
        
        Args:
            electrode_channels (list): List of electrode channel indices
            timestamp_channel (int): Timestamp channel index
            
        Returns:
            list: Combined list of all channel indices
        """
        all_channels = list(electrode_channels)
        
        if timestamp_channel is not None and timestamp_channel not in electrode_channels:
            all_channels.append(timestamp_channel)
            
        return all_channels

    def validate_consecutive_data(self, board_data_chunk, channel_idx=None):
        """
        Validate that data values on a specific channel are consecutive.
        This is useful for testing with synthetic data where we expect consecutive values.
        
        Args:
            board_data_chunk: Raw board data chunk (RawBoardDataWithKeys wrapper)
            channel_idx: Channel index to validate (defaults to self.validation_channel)
            
        Returns:
            tuple: (is_valid, message)
            - is_valid: True if data is consecutive, False otherwise
            - message: Description of any validation failure
        """
        if not self.validate_consecutive_values:
            return True, "Validation disabled"
            
        if channel_idx is None:
            channel_idx = self.validation_channel
            
        # Adjust for CSV structure where first column is not EEG data
        # The first EEG channel is actually at index 1 in the CSV
        adjusted_channel_idx = channel_idx + 1
        
        if adjusted_channel_idx >= board_data_chunk.shape[0]:
            return False, f"Channel index {adjusted_channel_idx} out of range"
        channel_data = board_data_chunk.get_by_key(adjusted_channel_idx)
        
        # For the first validation, just store the last value
        if self.last_validated_value_for_consecutive_data_validation is None:
            self.last_validated_value_for_consecutive_data_validation = channel_data[-1]
            return True, "First validation - stored last value"
            
        # Check if the new data starts where the last data ended
        expected_next_value = self.last_validated_value_for_consecutive_data_validation + 1
        if channel_data[0] != expected_next_value:
            return False, f"Non-consecutive data detected. Expected {expected_next_value}, got {channel_data[0]}"
            
        # Check if all values in the chunk are consecutive
        for i in range(1, len(channel_data)):
            if channel_data[i] != channel_data[i-1] + 1:
                return False, f"Non-consecutive data within chunk at index {i}. Expected {channel_data[i-1] + 1}, got {channel_data[i]}"
                
        # Update last validated value
        self.last_validated_value_for_consecutive_data_validation = channel_data[-1]
        return True, "Data validated successfully"

    def validate_data(self, board_data_chunk):
        """Validate incoming board data format and values.
        
        Args:
            board_data_chunk: Raw board data (RawBoardDataWithKeys wrapper). Must be in (n_channels, n_samples) format.
            
        Returns:
            bool: True if data is valid, False if validation failed
            
        Raises:
            ValueError: If data shape is invalid
            Exception: If consecutive value validation fails (when enabled)
        """
        # Validate data values 
        if np.any(np.isnan(board_data_chunk.data)) or np.any(np.isinf(board_data_chunk.data)):
            logging.warning("Data contains NaN or infinite values!")
            return False
        # Validate consecutive values if enabled
        if self.validate_consecutive_values:
            is_valid, message = self.validate_consecutive_data(board_data_chunk)
            if not is_valid:
                raise Exception(f"Consecutive value validation failed: {message}")
        # Validate data shape: must be (n_channels, n_samples)
        total_channels = self.board_shim.get_num_rows(self.board_shim.get_board_id())
        if board_data_chunk.shape[0] != total_channels:
            raise ValueError(f"Expected data in (n_channels, n_samples) format with {total_channels} channels, got shape {board_data_chunk.shape}")
        return True


    def queue_data_for_csv_write(self, new_data, is_initial=False):
        """Queue new data for CSV writing and handle buffer management."""
        self.csv_manager.queue_data_for_csv_write(new_data, is_initial)

    def validate_epoch_gaps(self, buffer_id, epoch_start_idx_abs, epoch_end_idx_abs):
        """Validate the epoch has no gaps
        
        Args:
            buffer_id: ID of the buffer being validated
            epoch_start_idx_abs: Absolute start index of the epoch in the total streamed data
            epoch_end_idx_abs: Absolute end index of the epoch in the total streamed data
            
        Returns:
            tuple: (has_gap, gap_size)
            - has_gap: True if a gap was detected
            - gap_size: Size of the gap if one was detected, otherwise 0
        """
        # Get timestamp data
        timestamp_data = self.etd_buffer_manager._get_timestamps()

        # Adjust the epoch indices to account for the buffer delay
        epoch_start_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(epoch_start_idx_abs)
        epoch_end_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(epoch_end_idx_abs)



        # Use GapHandler to validate gaps
        has_gap, gap_size = self.gap_handler.validate_epoch_gaps(
            timestamps=timestamp_data,
            epoch_start_idx_rel=epoch_start_idx_rel,
            epoch_end_idx_rel=epoch_end_idx_rel
        )
        
        return has_gap, gap_size

    def add_sleep_stage_to_csv_buffer(self, sleep_stage, next_buffer_id, epoch_end_idx_abs):
        """Add the sleep stage and buffer ID with timestamps to the sleep stage CSV
        
        Args:
            sleep_stage: The predicted sleep stage
            next_buffer_id: The ID of the buffer that processed this epoch
            epoch_end_idx_abs: Absolute end index of the epoch in the total streamed data
        """
        # Calculate timestamps for this epoch
        epoch_start_idx_abs = epoch_end_idx_abs - self.points_per_epoch
        
        # Get timestamps from the data - convert absolute indices to relative first
        timestamp_data = self.etd_buffer_manager._get_timestamps()
        epoch_start_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(epoch_start_idx_abs, to_etd=True)
        epoch_end_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(epoch_end_idx_abs - 1, to_etd=True)  # epoch_end_idx_abs is exclusive, so use -1
        timestamp_start = timestamp_data[epoch_start_idx_rel]
        timestamp_end = timestamp_data[epoch_end_idx_rel]

        
        # Use the new method signature with timestamps
        ## Needs absolute indices for the timestamps
        self.csv_manager.add_sleep_stage_to_sleep_stage_csv(
            float(sleep_stage[0]), 
            float(next_buffer_id), 
            float(timestamp_start), 
            float(timestamp_end)
        )

    def _get_total_data_points_etd(self):
        """Get the total number of data points currently in the electrode_and_timestamp_data buffer.
        
        Returns:
            int: The number of data points across all channels (they all have the same length).
        """
        return self.etd_buffer_manager._get_total_data_points()

    def _get_data_point_delay_for_buffer(self, buffer_id):
        """Calculate the data point delay in the round robin for a given buffer.
        
        Args:
            buffer_id: The ID of the buffer (0-5)
            
        Returns:
            int: Number of data points of delay for this buffer
        """
        return buffer_id * self.points_per_step

    def _get_next_epoch_indices(self, buffer_id):
        """Get the start and end indices for a buffers next epoch based on the last processed epoch plus the points per epoch.
        
        Args:
            buffer_id: The ID of the buffer to get indices for
            
        Returns:
            tuple: (epoch_start_idx_abs, epoch_end_idx_abs)
            - epoch_start_idx_abs: The absolute starting index for the epoch in the total streamed data
            - epoch_end_idx_abs: The absolute ending index for the epoch in the total streamed data
        """
        buffer_start_offset = self._get_data_point_delay_for_buffer(buffer_id)
        last_epoch_tuple = None
        last_epoch_start_idx_abs = None
        
        try:
            last_epoch_tuple = self.last_processed_epoch_per_buffer[buffer_id]
            if last_epoch_tuple is not None:
                last_epoch_start_idx_abs = last_epoch_tuple[0]  # Extract start index from tuple
        except (IndexError, KeyError, TypeError):
            pass

        # If this is the first epoch for this buffer, start at the buffer's offset
        if last_epoch_start_idx_abs is None:
            epoch_start_idx_abs = buffer_start_offset
        else:
            # Otherwise, start after the last processed epoch
            epoch_start_idx_abs = last_epoch_start_idx_abs + self.points_per_epoch

        epoch_end_idx_abs = epoch_start_idx_abs + self.points_per_epoch
        
        return epoch_start_idx_abs, epoch_end_idx_abs

    def _has_minimum_points_to_start_processing(self, epoch_end_idx_abs, total_etd_points, buffer_data_point_delay):
        """Check if we have the minimum required points to start processing epochs at this buffer position.
        
        This is the basic requirement that must be met before we can process ANY epochs
        (both initial and subsequent) at this buffer's position.
        
        Args:
            epoch_end_idx_abs: Absolute end index of the proposed epoch in the total streamed data
            total_etd_points: Total number of data points available
            buffer_data_point_delay: Delay points for this buffer
            
        Returns:
            bool: True if we have enough points to start processing, False otherwise
        """
        # Use total_streamed_samples for absolute index comparison since epoch_end_idx_abs is absolute
        total_streamed_samples = self.etd_buffer_manager.total_streamed_samples
        return (epoch_end_idx_abs <= total_streamed_samples and 
                total_etd_points >= buffer_data_point_delay)

    def _is_first_epoch_in_round_robin(self, buffer_id):
        """Check if this would be the first epoch processed in the round robin.
        
        Args:
            buffer_id: The ID of the buffer to check
            
        Returns:
            bool: True if no epochs have been processed yet in any buffer
        """
        has_processed_epochs = (self.last_processed_epoch_per_buffer[buffer_id] is not None or 
                              any(epoch is not None for epoch in self.last_processed_epoch_per_buffer))
        return not has_processed_epochs

    def _has_enough_delay_since_last_epoch(self):
        """Check if enough time has passed since the last epoch was processed.
        
        This check ensures we maintain proper spacing between epoch processing
        in the round-robin system.
        
        Returns:
            bool: True if enough time has passed since last epoch
        """
        last_etd_timestamp = self.etd_buffer_manager._get_timestamps()[-1]
        last_epoch = self.last_processed_epoch_per_buffer[self.last_processed_buffer]
        last_epoch_end_ind_abs = last_epoch[1]
        last_epoch_end_ind_rel = self.etd_buffer_manager._adjust_index_with_offset(last_epoch_end_ind_abs, to_etd=True)
        last_epoch_timestamp = self.etd_buffer_manager._get_timestamps()[last_epoch_end_ind_rel]
        return last_etd_timestamp - last_epoch_timestamp >= self.round_robin_delay

    def _calculate_next_buffer_id_to_process(self):
        """Calculate which buffer should be processed next in the round-robin sequence.
        
        Returns:
            int: The ID of the next buffer to process (0-5)
        """
        return (self.last_processed_buffer + 1) % 6

    def _validate_epoch_availability(self, buffer_id, epoch_end_idx_abs, total_etd_points, buffer_data_point_delay):
        """Validate if an epoch is available for processing.
        
        Args:
            buffer_id: The ID of the buffer to check
            epoch_end_idx_abs: Absolute end index of the proposed epoch in the total streamed data
            total_etd_points: Total number of data points available
            buffer_data_point_delay: Delay points for this buffer
            
        Returns:
            tuple: (is_valid, reason)
                - is_valid: True if epoch is valid for processing
                - reason: Explanation if not valid, None if valid
        """
        # First check - Always verify we have minimum required points to start processing
        if not self._has_minimum_points_to_start_processing(epoch_end_idx_abs, total_etd_points, buffer_data_point_delay):
            return False, "Need more samples to form complete epoch"
            
        # If this is the first epoch, we only need the minimum points check
        if self._is_first_epoch_in_round_robin(buffer_id):
            return True, None
            
        # Additional check for subsequent epochs - enforce minimum delay between epochs
        if not self._has_enough_delay_since_last_epoch():
            return False, "Need more samples to form complete epoch"
            
        return True, None

    def next_available_epoch_on_round_robin_buffer(self, buffer_id):
        """Return the next available epoch on the round robin buffer.
        
        For an epoch to be available:
        - We need enough points to cover both buffer delay and a full epoch
        - We need enough time to have passed since the last epoch on any buffer
        
        Args:
            buffer_id: The ID of the round robin buffer to check (0-5)
            
        Returns:
            tuple: (can_process, reason, epoch_start_idx_abs, epoch_end_idx_abs)
                - can_process: True if we can process this buffer
                - reason: Explanation if we can't process, empty string if we can
                - epoch_start_idx_abs: Start index of next epoch if can_process, else None
                - epoch_end_idx_abs: End index of next epoch if can_process, else None
        """
        # Get required values
        buffer_data_point_delay = self._get_data_point_delay_for_buffer(buffer_id)
        total_etd_points = self._get_total_data_points_etd()
        epoch_start_idx_abs, epoch_end_idx_abs = self._get_next_epoch_indices(buffer_id)
        
        # Validate epoch availability
        is_valid, reason = self._validate_epoch_availability(
            buffer_id, 
            epoch_end_idx_abs, 
            total_etd_points, 
            buffer_data_point_delay
        )
        
        # Return combined result
        return (
            is_valid,
            reason if not is_valid else None,
            epoch_start_idx_abs if is_valid else None,
            epoch_end_idx_abs if is_valid else None
        )

    def manage_epoch(self, buffer_id, epoch_start_idx_abs, epoch_end_idx_abs):
        """Validate and process a specified epoch on a specified buffer."""

        # validate that we can process the epoch
        has_gap, gap_size = self.validate_epoch_gaps(buffer_id, epoch_start_idx_abs, epoch_end_idx_abs)

        if has_gap:
            # Handle the gap
            try:
                prev_timestamp = self.etd_buffer_manager._get_timestamps()[epoch_start_idx_abs-1]
                self.handle_gap(
                    prev_timestamp=prev_timestamp,
                    gap_size=gap_size, buffer_id=buffer_id
                )
            except IndexError as e: #added due to myterious error we saw previously when straming data
                timestamps = self.etd_buffer_manager._get_timestamps()
                self.logger.error(f"IndexError in gap handling - Debug info:")
                self.logger.error(f"  - epoch_start_idx_abs: {epoch_start_idx_abs}")
                self.logger.error(f"  - epoch_end_idx_abs: {epoch_end_idx_abs}")
                self.logger.error(f"  - buffer_id: {buffer_id}")
                self.logger.error(f"  - gap_size: {gap_size}")
                self.logger.error(f"  - timestamps length: {len(timestamps)}")
                self.logger.error(f"  - trying to access index: {epoch_start_idx_abs-1}")
                self.logger.error(f"  - timestamps array: {timestamps[:10] if len(timestamps) > 0 else 'EMPTY'}")
                self.logger.error(f"  - buffer offsets: {[self._get_data_point_delay_for_buffer(i) for i in range(6)]}")
                self.logger.error(f"  - total ETD buffer size: {len(self.etd_buffer_manager.get_buffer_data())}")
                raise e  # Re-raise to maintain original behavior
       
        # Update buffer status
        self.last_processed_epoch_per_buffer[buffer_id] = (epoch_start_idx_abs, epoch_end_idx_abs)
        self.epochs_processed_count_per_buffer[buffer_id] += 1
        self.last_processed_buffer = buffer_id  

        if has_gap:
            return

        # Process the epoch
        sleep_stage = self._process_epoch(start_idx_abs=epoch_start_idx_abs, end_idx_abs=epoch_end_idx_abs, buffer_id=buffer_id)
        
        # Add sleep stage to CSV
        self.add_sleep_stage_to_csv_buffer(sleep_stage, buffer_id, epoch_end_idx_abs)

    def _process_epoch(self, start_idx_abs, end_idx_abs, buffer_id):
        """Handle the data for a specified epoch on a specified buffer which has valid data.
        
        Args:
            start_idx_abs: Absolute start index of the epoch in the total streamed data
            end_idx_abs: Absolute end index of the epoch in the total streamed data
            buffer_id: The ID of the buffer to process
            
        Returns:
            np.ndarray: Array containing the predicted sleep stage
        """
        
        print(f"\nProcessing buffer {buffer_id}")
        print(f"Epoch range: {start_idx_abs} to {end_idx_abs}")
        print(f"Buffer {buffer_id}: Epoch range: {start_idx_abs * self.expected_interval} to {end_idx_abs * self.expected_interval} seconds")
        
        # Convert absolute indices to relative indices for buffer access
        start_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(start_idx_abs, to_etd=True)
        end_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(end_idx_abs, to_etd=True)
        
        # Validate epoch duration
        self._validate_epoch_duration(start_idx_abs, end_idx_abs)
        
        # Extract EXACTLY points_per_epoch data points from the correct slice
        # Get electrode board keys (exclude timestamp channel)
        electrode_board_keys = [key for key in self.etd_buffer_manager.electrode_and_timestamp_board_keys 
                               if key != self.etd_buffer_manager.timestamp_board_key]
        
        epoch_data = np.array([
            self.etd_buffer_manager.electrode_and_timestamp_data_keyed.get_by_key(board_key)[start_idx_rel:end_idx_rel]
            for board_key in electrode_board_keys
        ])
        num_electrode_channels = len(electrode_board_keys)
        
        # Verify we have exactly the right number of points
        assert epoch_data.shape[1] == self.points_per_epoch, f"Expected {self.points_per_epoch} points, got {epoch_data.shape[1]}"
        
        # Get the timestamp data for this epoch
        timestamp_data = self.etd_buffer_manager._get_timestamps()[start_idx_rel:end_idx_rel]
        epoch_start_time = timestamp_data[0]  # First timestamp in the epoch
        
        # Get index combinations for EEG and EOG channels
        # 
        # CHANNEL MAPPING: epoch_data_all_electrodes_on_board is always 16 channels from physical OpenBCI channels 1-16
        # Array index = physical_channel - 1 (0-based indexing)
        # - epoch_data_all_electrodes_on_board[0] = physical channel 1
        # - epoch_data_all_electrodes_on_board[10] = physical channel 11 (R-LEOG) 
        # - epoch_data_all_electrodes_on_board[11] = physical channel 12 (L-LEOG)
        #
        # The montage type doesn't change this mapping - it only defines which channels are used.
        
        channel_types = self.montage.get_channel_types()
        
        # Create structured channel mapping for epoch data using actual board keys
        epoch_data_channel_mapping = [
            ChannelIndexMapping(board_position=board_key)
            for board_key in electrode_board_keys
        ]
        
        # Wrap epoch data with structured mapping
        epoch_data_keyed = NumPyDataWithBrainFlowDataKey(
            data=epoch_data,
            channel_mapping=epoch_data_channel_mapping
        )
        
        if all(ch_type == "EOG" for ch_type in channel_types):
            # EOG-only montage: no EEG channels, use both EOG channels
            # Get data by board positions 11, 12 (physical channels 11, 12)
            eeg_combo_pk = []        # No EEG channels
            eog_combo_pk = [10, 11]  # Keep as array indices for backward compatibility
        else:
            # Default for minimal_sleep_montage and other montages
            # Get data by board positions 1, 2, 3 and 11 (physical channels 1, 2, 3, 11)
            eeg_combo_pk = [1, 2, 3]  # Keep as array indices for backward compatibility
            eog_combo_pk = [10]       # Keep as array indices for backward compatibility
        
        # Validate that our hardcoded indices match the expected channel types
        self.montage.validate_channel_indices_combination_types(eeg_combo_pk, eog_combo_pk)
        
        # IMPORTANT: The GSSC was mostly trained on EEG channels C3, C4, F3, F4, and using other channels
        # is unlikely to improve accuracy and could even make accuracy worse. For EOG, the GSSC was trained
        # on the HEOG channel (left EOG - right EOG differential), and seems to also perform well with left
        # and/or right EOG alone (without the subtraction). Effectiveness of using VEOG is unknown and not recommended.
        index_combinations = self.signal_processor.get_index_combinations(eeg_combo_pk, eog_combo_pk)
        
        # Get sleep stage prediction using SignalProcessor
        predicted_class, class_probs, new_hidden_states = self.signal_processor.predict_sleep_stage(
            epoch_data_keyed,
            index_combinations,
            self.buffer_hidden_states[buffer_id]
        )
        
        # Increment epochs scored counter
        self.epochs_scored += 1
        print(f"Sleep stage: {self.visualizer.get_sleep_stage_text(predicted_class)}")
        print(f"Total epochs scored: {self.epochs_scored}")
        
        # Emit event if event dispatcher is available
        if self.event_dispatcher is not None:
            self.event_dispatcher.emit_sleep_stage_event(
                sleep_stage=predicted_class,
                timestamp=epoch_start_time,
                class_probabilities=class_probs,
                epoch_data=epoch_data_keyed,
                buffer_id=buffer_id
            )
        
        # Update visualization using Visualizer
        time_offset = start_idx_abs / self.sampling_rate
        self.visualizer.plot_polysomnograph(
            epoch_data_keyed, 
            self.sampling_rate, 
            predicted_class, 
            time_offset, 
            epoch_start_time
        )
        
        self.buffer_hidden_states[buffer_id] = new_hidden_states
        return np.array([predicted_class])  # Keep return format consistent

    def save_to_csv(self, output_path):
        """Save all remaining data to sleep stage CSV file and main CSV file"""
        self.output_csv_path = output_path
        # Set the CSV paths before saving
        self.csv_manager.main_csv_path = output_path
        if self.csv_manager.sleep_stage_csv_path is None:
            # Use default convention for sleep stage path
            self.csv_manager.sleep_stage_csv_path = self.csv_manager._get_default_sleep_stage_path(output_path)
        
        # Debug logging
        print(f"\n=== Debug: DataManager.save_to_csv ===")
        print(f"Total points collected: {self.etd_buffer_manager.total_streamed_samples}")
        print(f"CSVManager main_buffer_size: {self.csv_manager.main_buffer_size}")
        print(f"CSVManager main_csv_buffer length: {len(self.csv_manager.main_csv_buffer)}")
        
        # Save all data but don't cleanup yet (for validation)
        result = self.csv_manager.save_all_data()
        
        # Keep the path for validation, cleanup will be done in DataManager.cleanup()
        return result

    def validate_saved_csv(self, original_csv_path, output_csv_path):
        """Validate that the saved CSV matches the original format exactly, ignoring sleep stage and buffer ID columns"""
        result = compare_csv_files(output_csv_path, original_csv_path, logger=logger)
        if not result.matches:
            raise CSVDataError(f"CSV validation failed: {result.error_message}")
        return result.matches

    def detect_gap(self, timestamps, prev_timestamp):
        """
        Detect if there is a gap in the timestamps.
        Checks both between chunks and within the current chunk.
        
        Args:
            timestamps: Current timestamps array
            prev_timestamp: Previous timestamp to compare against
            
        Returns:
            tuple: (has_gap, gap_size, gap_start_idx, gap_end_idx)
            - has_gap: True if a gap was detected
            - gap_size: Size of the largest gap found
            - gap_start_idx: Start index of the gap (or None if no gap)
            - gap_end_idx: End index of the gap (or None if no gap)
        """
        # Convert timestamps to numpy array if not already
        timestamps = np.array(timestamps)
        return self.gap_handler.detect_gap(timestamps=timestamps, prev_timestamp=prev_timestamp)
        
        # OLD IMPLEMENTATION:
        # """
        # has_gap = False
        # max_gap = 0
        # gap_start_idx = None
        # gap_end_idx = None
        # 
        # # Check gap between chunks if we have a previous timestamp
        # if prev_timestamp is not None:
        #     between_chunks_gap = timestamps[0] - prev_timestamp - self.expected_interval
        #     if abs(between_chunks_gap) >= self.gap_threshold:
        #         has_gap = True
        #         max_gap = between_chunks_gap
        #         gap_start_idx = -1  # -1 indicates gap is between chunks
        #         gap_end_idx = 0
        # 
        # # Check gaps within the chunk
        # for i in range(1, len(timestamps)):
        #     interval_deviation = timestamps[i] - timestamps[i-1] - self.expected_interval
        #     if abs(interval_deviation) >= self.gap_threshold:
        #         has_gap = True
        #         if abs(interval_deviation) > abs(max_gap):
        #             max_gap = interval_deviation
        #             gap_start_idx = i-1
        #             gap_end_idx = i
        # 
        # return has_gap, max_gap, gap_start_idx, gap_end_idx
        # """
        
        # Use GapHandler to detect gaps

    def handle_gap(self, prev_timestamp, gap_size, buffer_id):
        """
        Handle a detected gap by resetting the appropriate buffer.
        
        Args:
            prev_timestamp: Timestamp where gap was detected
            gap_size: Size of the gap in seconds
            buffer_id: Buffer ID to reset
        """
        self.reset_buffer_states(buffer_id, gap_size)

    def reset_buffer_states(self, buffer_id, gap_size):
        """
        Reset the hidden states and buffer indices for the affected buffer.
        
        Args:
            buffer_id: Buffer ID to reset
            gap_size: Size of the gap (for logging)
        """
        if buffer_id is not None:
            print(f"\nLarge gap detected ({gap_size:.2f}s): Resetting buffer {buffer_id}")
            
            # Reset hidden states
            self.buffer_hidden_states[buffer_id] = [
                torch.zeros(10, 1, 256) for _ in range(7)
            ]
            
            print(f"Buffer {buffer_id} reset complete - hidden states cleared")

    def cleanup(self):
        """Clean up resources and reset state"""
        try:
            # Log final epoch count before cleanup
            print(f"\nFinal epoch count: {self.epochs_scored} epochs were scored")
            
            # Clean up CSVManager - reset paths since we're done
            if hasattr(self, 'csv_manager'):
                self.csv_manager.cleanup()
            
            # Reset state variables
            self.last_processed_buffer = -1
            self.current_epoch_start_time = None
            self.last_validated_value_for_consecutive_data_validation = None
            self.output_csv_path = None
            self.last_saved_timestamp = None
            
            # Clear data buffers
            self.etd_buffer_manager.electrode_and_timestamp_data_keyed.clear_all()
            self.last_processed_epoch_per_buffer = [None] * 6
            self.epochs_processed_count_per_buffer = [0] * 6
            
            # Reset hidden states
            self.buffer_hidden_states = [
                [torch.zeros(10, 1, 256) for _ in range(7)]  # 7 hidden states for 7 combinations
                for _ in range(6)  # 6 buffers (0s to 25s in 5s steps)
            ]
            
            # Clean up GapHandler
            if hasattr(self, 'gap_handler'):
                self.gap_handler.cleanup()
            
            logging.info("DataManager cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during DataManager cleanup: {str(e)}")
            raise

    def _validate_epoch_duration(self, start_idx_abs: int, end_idx_abs: int) -> None:
        """Validate that epoch duration is exactly 30 seconds.
        
        Args:
            start_idx_abs: Start index absolute in the total streamed data
            end_idx_abs: End index absolute in the total streamed data
            
        Raises:
            ValueError: If epoch duration is not 30 seconds within tolerance
        """
        # Convert absolute indices to relative indices for buffer access
        start_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(start_idx_abs, to_etd=True)
        end_idx_rel = self.etd_buffer_manager._adjust_index_with_offset(end_idx_abs, to_etd=True)
        
        timestamp_data = self.etd_buffer_manager._get_timestamps()
        epoch_start_timestamp = timestamp_data[start_idx_rel]
        epoch_end_timestamp = timestamp_data[end_idx_rel - 1]  # -1 because end_idx_rel is exclusive
        
        epoch_duration = epoch_end_timestamp - epoch_start_timestamp
        expected_duration = 30.0
        tolerance = 0.1  # Allow 100ms tolerance for floating point precision
        
        if abs(epoch_duration - expected_duration) > tolerance:
            raise ValueError(
                f"Epoch duration validation failed: Expected {expected_duration}s, "
                f"got {epoch_duration:.6f}s (difference: {epoch_duration - expected_duration:.6f}s). "
                f"Start: {epoch_start_timestamp:.6f}, End: {epoch_end_timestamp:.6f}, "
                f"Indices: {start_idx_abs}-{end_idx_abs} (abs), {start_idx_rel}-{end_idx_rel} (rel)"
            )








