# the cyton_realtime_round_robin.py + we are added accurate time handling
# realtime inference with round robin buffer with continuous data (eog data: [1, 2, 3, 4, 5 ...] ) for testing purposes
# displays data and sleep stage predictions updated every 5 seconds
# seems to work now

#todo: make sure it works for cyton and daisy
#todo: make sure it works for non consecutive numbers. we used consecutive data for testing
#todo: scale graph better
#todo: save the data stream to a csv file
import logging
import time
import os
import matplotlib.pyplot as plt
import datetime

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import brainflow
import numpy as np
from gssc.infer import ArrayInfer
import mne

import warnings

from gssc.utils import permute_sigs, prepare_inst, epo_arr_zscore, loudest_vote

import torch

import torch.nn.functional as F

import pandas as pd

# Add at the top of the file, after imports
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Add this as a global variable at the top of the file, after the imports
global_fig = None
global_axes = None

# Add at the top of the file, after imports
DEBUG_VERBOSE = False  # Set to True only when you need detailed debugging

# Add after imports
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)  # Suppress font manager debug messages

# Add at the top with other globals
global_recording_start_time = None

class SignalProcessor:
    """Handles signal processing and sleep stage prediction"""
    def __init__(self, use_cuda=False, gpu_idx=None):
        self.infer = ArrayInfer(
            net=None,  # Use default network
            con_net=None,  # Use default context network
            use_cuda=use_cuda,
            gpu_idx=gpu_idx
        )
        
    def resample_tensor(self, data, target_length):
        """Resample a tensor to a target length using interpolation"""
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        resampled = F.interpolate(
            data, 
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        if len(data.shape) == 2:
            resampled = resampled.squeeze(0)
            
        return resampled
    
    def make_combo_dictionary(self, epoch_data, eeg_index, eog_index):
        """Create input dictionary for EEG and EOG data"""
        input_dict = {}
        epoch_tensor = (torch.tensor(epoch_data, dtype=torch.float32) 
                       if not isinstance(epoch_data, torch.Tensor) 
                       else epoch_data.float())
        
        if eeg_index is not None:
            eeg_data = epoch_tensor[eeg_index].unsqueeze(0).unsqueeze(0)
            input_dict['eeg'] = eeg_data
        
        if eog_index is not None:
            eog_data = epoch_tensor[eog_index].unsqueeze(0).unsqueeze(0)
            input_dict['eog'] = eog_data
        
        return input_dict
    
    def get_index_combinations(self, eeg_indices, eog_indices):
        """Generate all possible combinations of EEG and EOG indices"""
        combinations = []
        
        # Add EEG-EOG pairs
        for eeg_idx in eeg_indices:
            for eog_idx in eog_indices:
                combinations.append([eeg_idx, eog_idx])
        
        # Add EEG only combinations
        for eeg_idx in eeg_indices:
            combinations.append([eeg_idx, None])
        
        # Add EOG only combinations
        for eog_idx in eog_indices:
            combinations.append([None, eog_idx])
            
        return combinations
    
    def prepare_input_data(self, epoch_data):
        """Prepare input data for prediction"""
        # Hardcoded indices for now - could be made configurable
        index_combinations = self.get_index_combinations([0, 1, 2], [3])
        
        # Create input dictionaries
        input_dict_list = []
        for eeg_idx, eog_idx in index_combinations:
            input_dict = self.make_combo_dictionary(epoch_data, eeg_idx, eog_idx)
            input_dict_list.append(input_dict)
        
        # Resample all inputs to 2560
        resampled_dict_list = []
        for input_dict in input_dict_list:
            new_dict = {}
            if 'eeg' in input_dict:
                new_dict['eeg'] = self.resample_tensor(input_dict['eeg'], 2560)
            if 'eog' in input_dict:
                new_dict['eog'] = self.resample_tensor(input_dict['eog'], 2560)
            resampled_dict_list.append(new_dict)
            
        return resampled_dict_list
    
    def predict_sleep_stage(self, epoch_data, hidden_states):
        """Predict sleep stage from epoch data"""
        # Prepare input data
        input_dict_list = self.prepare_input_data(epoch_data)
        
        # Get predictions for each combination
        results = []
        for i, input_dict in enumerate(input_dict_list):
            logits, res_logits, hidden_state = self.infer.infer(input_dict, hidden_states[i])
            results.append([logits, res_logits, hidden_state])
        
        # Combine logits
        all_combo_logits = np.stack([
            results[i][0].numpy() for i in range(len(results))
        ])
        
        # Get final prediction
        final_predicted_class = loudest_vote(all_combo_logits)
        new_hidden_states = [result[2] for result in results]
        
        return final_predicted_class, new_hidden_states

class Visualizer:
    """Handles visualization of polysomnograph data and sleep stages"""
    def __init__(self, seconds_per_epoch=30):
        self.fig = None
        self.axes = None
        self.recording_start_time = None
        self.channel_labels = ['EEG Ch1', 'EEG Ch2', 'EEG Ch3', 'EOG']
        self.seconds_per_epoch = seconds_per_epoch
        
    def init_polysomnograph(self):
        """Initialize the polysomnograph figure and axes"""
        if self.fig is None:
            plt.ion()  # Turn on interactive mode
            self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
            
            # Setup axes
            for ax, label in zip(self.axes, self.channel_labels):
                ax.set_ylabel(label)
                ax.grid(True)
            
            self.axes[-1].set_xlabel('Time (seconds)')
            
            # Add extra space at the top for the title
            plt.subplots_adjust(top=0.85)
        
        return self.fig, self.axes
    
    @staticmethod
    def get_sleep_stage_text(sleep_stage):
        """Convert sleep stage number to text representation"""
        stages = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        return stages.get(sleep_stage, 'Unknown')
    
    def plot_polysomnograph(self, epoch_data, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
        """Update polysomnograph plot with new data"""
        self.init_polysomnograph()  # Ensure figure exists
        
        # Create time axis with offset
        time_axis = np.arange(epoch_data.shape[1]) / sampling_rate + time_offset
        
        # Calculate elapsed time
        elapsed_seconds = (epoch_start_time - self.recording_start_time 
                         if self.recording_start_time is not None and epoch_start_time is not None 
                         else time_offset)
        
        # Format time string
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        relative_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Update title
        title_text = (f'Sleep Stage: {self.get_sleep_stage_text(sleep_stage)}\n'
                     f'Time from Start: {relative_time_str}\n')
        
        # Plot each channel
        for ax, data, label in zip(self.axes, epoch_data, self.channel_labels):
            ax.clear()  # Clear previous data
            ax.plot(time_axis, data, 'b-', linewidth=0.5)
            ax.set_ylabel(label)
            ax.grid(True)
            
            # Set y-axis limits
            y_range = np.percentile(data, [5, 95])
            margin = (y_range[1] - y_range[0]) * 0.2
            ax.set_ylim(y_range[0] - margin, y_range[1] + margin)
        
        # Add title text after clearing and plotting
        self.axes[0].text(0.5, 1.25, title_text,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.axes[0].transAxes,
                         fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.8, pad=0.5))
        
        # Update x-axis label
        self.axes[-1].set_xlabel('Time from Start (seconds)')
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



class DataAcquisition:
    """Handles board setup and data collection"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.board_shim = None
        self.sampling_rate = None
        self.buffer_size = 450000
        self.points_collected = 0
        self.recording_start_time = None  # Add this to track start time
        self.last_chunk_last_timestamp = None
        self.file_data = None
        self.current_position = 0
        self.buffer_manager = None
        self.master_board_id = BoardIds.CYTON_DAISY_BOARD  # Add master board ID
        self.timestamp_channel = None  # Will be set in setup_board
        self.gap_threshold = 2.0  # Add gap threshold in seconds
    
    def set_buffer_manager(self, buffer_manager):
        self.buffer_manager = buffer_manager
    
    def setup_board(self):
        """Initialize and setup the board for data collection"""
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

        params = BrainFlowInputParams()
        params.board_id = BoardIds.PLAYBACK_FILE_BOARD
        params.master_board = self.master_board_id  # Use class variable
        params.file = self.file_path
        params.playback_file_max_count = 1
        params.playback_speed = 1
        params.playback_file_offset = 0

        self.board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
        
        try:
            self.board_shim.prepare_session()
            # Get sampling rate and timestamp channel from master board, not playback board
            self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.master_board_id)
            
            print(f"\nBoard Configuration:")
            print(f"Master board: {self.master_board_id}")
            print(f"Timestamp channel: {self.timestamp_channel}")
            print(f"Sampling rate: {self.sampling_rate}")
            
            self.board_shim.config_board("old_timestamps")

            # Load the entire file data at initialization
            self.file_data = pd.read_csv(self.file_path, sep='\t', dtype=float)
            print(f"Loaded file with {len(self.file_data)} samples")
            
            return self.board_shim
            
        except Exception as e:
            logging.error(f"Failed to setup board: {str(e)}")
            raise

    def start_stream(self):
        """Start the data stream"""
        if not self.board_shim:
            raise RuntimeError("Board not initialized. Call setup_board first.")
        
        try:
            print("\nStarting data stream:")
            print(f"- Buffer size: {self.buffer_size}")
            print(f"- Sampling rate: {self.sampling_rate}")
            print(f"- Timestamp channel: {self.timestamp_channel}")
            
            self.board_shim.start_stream(self.buffer_size)
            
            # Initialize recording_start_time as None - we'll set it when we get the first data packet
            self.recording_start_time = None
            
            # Wait briefly for stream to start
            time.sleep(0.1)
            
            initial_count = self.board_shim.get_board_data_count()

            
            return initial_count
        except Exception as e:
            logging.error(f"Failed to start stream: {str(e)}")
            raise

    def get_initial_data(self):
        """Get initial data from the board"""
        if not self.board_shim:
            raise RuntimeError("Board not initialized. Call setup_board first.")
        
        try:
            # Log data count before getting data
            data_count = self.board_shim.get_board_data_count()
            print(f"\nAttempting to get initial data:")
            print(f"- Available data count: {data_count}")
            
            if data_count > 0:
                # Get all available data
                initial_data = self.board_shim.get_board_data(data_count)

                # Log details about the retrieved data
                if initial_data.size <= 0:
                    print("- get_board_data returned empty array despite positive count!")
                else:
                    # For streaming, set recording_start_time to the first timestamp if not already set
                    if self.recording_start_time is None and self.timestamp_channel is not None:
                        self.recording_start_time = initial_data[self.timestamp_channel][0]
                        print(f"- Set recording start time to: {self.recording_start_time}")
                    
                return initial_data
            else:
                return np.array([])
            
        except Exception as e:
            logging.error(f"Failed to get initial data: {str(e)}")
            print(f"- Exception details: {str(e)}")
            print(f"- Exception type: {type(e)}")
            raise

    def get_channel_info(self):
        """Get information about board channels"""
        if not self.board_shim:
            raise RuntimeError("Board not initialized. Call setup_board first.")
            
        all_channels = self.board_shim.get_exg_channels(self.board_shim.get_board_id())
        timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
        all_channels_with_timestamp = list(all_channels)
        
        if timestamp_channel is not None and timestamp_channel not in all_channels:
            all_channels_with_timestamp.append(timestamp_channel)

        return all_channels, timestamp_channel, all_channels_with_timestamp

    def get_new_data(self):
        """Get new data from the board"""
        try:
            # Log data count before getting data
            data_count = self.board_shim.get_board_data_count()

            if data_count > 0:
                # Get all available data
                new_data = self.board_shim.get_board_data(data_count)

                if new_data.size > 0:
                    
                    if self.timestamp_channel is not None:
                        last_timestamp = new_data[self.timestamp_channel][-1]
                        

                        self.last_chunk_last_timestamp = last_timestamp
                    
                    return new_data
                else:
                    print("- get_board_data returned empty array despite positive count!")
                    return np.array([])
            else:
                return np.array([])
            
        except Exception as e:
            logging.error(f"Failed to get new data: {str(e)}")
            print(f"- Exception details: {str(e)}")
            raise

    def release(self):
        """Release the board session"""
        if self.board_shim and self.board_shim.is_prepared():
            try:
                self.board_shim.release_session()
                logging.info('Session released successfully')
            except Exception as e:
                logging.error(f"Failed to release session: {str(e)}")
                raise

class BufferManager:
    """Manages data buffers and their processing"""
    def __init__(self, board_shim, sampling_rate):
        self.board_shim = board_shim
        self.sampling_rate = sampling_rate
        self.seconds_per_epoch = 30
        self.seconds_per_step = 5
        # Buffer configuration
        self.points_per_epoch = self.seconds_per_epoch * sampling_rate  # 30 second epochs
        self.points_per_step = self.seconds_per_step * sampling_rate    # 5 second steps
        self.buffer_start = 0
        self.buffer_end = self.seconds_per_epoch
        self.buffer_step = self.seconds_per_step
        
        # Initialize channels and buffers
        self.electrode_channels, self.timestamp_channel, self.all_channels_with_timestamp = self._init_channels()
        self.all_previous_buffers_data = [[] for _ in range(len(self.all_channels_with_timestamp))]
        
        # Buffer tracking
        self.points_collected = 0
        self.last_processed_buffer = -1
        
        # Add timing control
        self.epoch_interval = self.buffer_step  # Process every buffer_step seconds
        
        # List of lists tracking where each buffer has started processing epochs
        # Example structure after processing some epochs:
        # [
        #     [0, 6000, 12000],      # Buffer 0 processed epochs starting at indices 0, 6000, 12000
        #     [1000, 7000, 13000],   # Buffer 1 processed epochs starting at indices 1000, 7000, 13000
        #     [2000, 8000, 14000],   # Buffer 2 processed epochs starting at indices 2000, 8000, 14000
        #     [],                    # Buffer 3 hasn't processed any epochs yet
        #     [],                    # Buffer 4 hasn't processed any epochs yet
        #     []                     # Buffer 5 hasn't processed any epochs yet
        # ]
        self.processed_epoch_start_indices = [[] for _ in range(6)]
        
        # Initialize hidden states for each buffer
        self.buffer_hidden_states = [
            [torch.zeros(10, 1, 256) for _ in range(7)]  # 7 hidden states for 7 combinations
            for _ in range(6)  # 6 buffers (0s to 25s in 5s steps)
        ]
        self.signal_processor = SignalProcessor()
        self.visualizer = Visualizer(self.seconds_per_epoch)
        self.expected_interval = 1.0 / sampling_rate
        self.timestamp_tolerance = self.expected_interval * 0.01  # 1% tolerance
        self.gap_threshold = 2.0  # Large gap threshold (seconds)
        self.interpolation_threshold = 0.1  # Maximum gap to interpolate (seconds)
        self.current_epoch_start_time = None
        self.buffer_timestamp_index = self.all_channels_with_timestamp.index(self.timestamp_channel)
        
        # Add validation settings
        self.validate_consecutive_values = False  # Set to True to enable consecutive value validation
        self.validation_channel = 0  # Channel to validate (default to first channel)
        self.last_validated_value = None  # Track last validated value
        self.saved_data = []
        self.output_csv_path = None
        self.last_saved_timestamp = None  # Track last saved timestamp to prevent duplicates

    def _init_channels(self):
        """Initialize channel information"""
        electrode_channels = self.board_shim.get_exg_channels(self.board_shim.get_board_id())
        board_timestamp_channel = self.board_shim.get_timestamp_channel(self.board_shim.get_board_id())
        all_channels_with_timestamp = list(electrode_channels)
        
        if board_timestamp_channel is not None and board_timestamp_channel not in electrode_channels:
            all_channels_with_timestamp.append(board_timestamp_channel)
            self.buffer_timestamp_index = len(all_channels_with_timestamp) - 1
        
        return electrode_channels, board_timestamp_channel, all_channels_with_timestamp

    def validate_consecutive_data(self, data, channel_idx=None):
        """
        Validate that data values on a specific channel are consecutive.
        This is useful for testing with synthetic data where we expect consecutive values.
        
        Args:
            data: New data chunk
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
            
        if adjusted_channel_idx >= len(data):
            return False, f"Channel index {adjusted_channel_idx} out of range"
            
        channel_data = data[adjusted_channel_idx]
        
        # For the first validation, just store the last value
        if self.last_validated_value is None:
            self.last_validated_value = channel_data[-1]
            return True, "First validation - stored last value"
            
        # Check if the new data starts where the last data ended
        expected_next_value = self.last_validated_value + 1
        if channel_data[0] != expected_next_value:
            return False, f"Non-consecutive data detected. Expected {expected_next_value}, got {channel_data[0]}"
            
        # Check if all values in the chunk are consecutive
        for i in range(1, len(channel_data)):
            if channel_data[i] != channel_data[i-1] + 1:
                return False, f"Non-consecutive data within chunk at index {i}. Expected {channel_data[i-1] + 1}, got {channel_data[i]}"
                
        # Update last validated value
        self.last_validated_value = channel_data[-1]
        return True, "Data validated successfully"

    def add_data(self, new_data, is_initial=False):
        """Add new data to the buffer"""
        # Validate data values
        if np.any(np.isnan(new_data)) or np.any(np.isinf(new_data)):
            logging.warning("Data contains NaN or infinite values!")
            return False
            
        # Validate consecutive values if enabled
        if self.validate_consecutive_values:
            is_valid, message = self.validate_consecutive_data(new_data)
            if not is_valid:
                #  throw an exception
                raise Exception(f"Consecutive value validation failed: {message}")
            
        # Update points collected
        self.points_collected += len(new_data[0])
        
        # Update all_previous_data
        if not is_initial:
            for i, channel in enumerate(self.all_channels_with_timestamp):
                self.all_previous_buffers_data[i].extend(new_data[channel].tolist())
        else:
            for i, channel in enumerate(self.all_channels_with_timestamp):
                self.all_previous_buffers_data[i] = new_data[channel].tolist()
                
        
        # Handle data saving - only save new data
        new_rows = new_data.T.tolist()
        if not is_initial:
            # For subsequent data, only save rows with timestamps we haven't seen
            if self.last_saved_timestamp is not None:
                new_rows = [row for row in new_rows if row[self.buffer_timestamp_index] > self.last_saved_timestamp]
            
            if new_rows:
                if not hasattr(self, 'saved_data'):
                    self.saved_data = new_rows
                else:
                    self.saved_data.extend(new_rows)
                self.last_saved_timestamp = new_rows[-1][self.buffer_timestamp_index]
        else:
            # For initial data, save everything and set last timestamp
            self.saved_data = new_rows
            self.last_saved_timestamp = new_rows[-1][self.buffer_timestamp_index]
                
        return True

    def validate_epoch_gaps(self, buffer_id, epoch_start_idx, epoch_end_idx):
        """Validate the epoch has no gaps
        
        Args:
            buffer_id: ID of the buffer being validated
            epoch_start_idx: Start index of the epoch
            epoch_end_idx: End index of the epoch
            
        Returns:
            tuple: ( has_gap, gap_size)
            - has_gap: True if a gap was detected
            - gap_size: Size of the gap if one was detected, otherwise 0
        """
            
        # Check for gaps in the timestamp data
        timestamp_data = self.all_previous_buffers_data[self.buffer_timestamp_index]
        has_gap, gap_size, gap_start_idx, gap_end_idx = self.detect_gap(
            timestamp_data[epoch_start_idx:epoch_end_idx],
            timestamp_data[epoch_start_idx-1] if epoch_start_idx > 0 else None
        )
        
        return has_gap, gap_size

    def _enough_data_to_process_epoch(self, buffer_id, epoch_end_idx):
        """Check if we have enough data to process the given buffer"""
        buffer_delay = buffer_id * self.points_per_step

        
        return (epoch_end_idx <= len(self.all_previous_buffers_data[0]) and 
                len(self.all_previous_buffers_data[0]) >= buffer_delay)
    
    def _has_enough_data_for_buffer(self, buffer_id):
        """Check if we have enough data points and time has passed for the specified buffer
        
        Args:
            buffer_id: The ID of the buffer to check
            
        Returns:
            tuple: (bool, str)
                - bool: True if we can process this buffer, False otherwise
                - str: Reason why we can't process if False, empty string if we can
        """
        # For the first buffer, we just need enough points for one epoch
        if buffer_id == 0:
            required_points = self.points_per_epoch
        else:
            # For other buffers, we need enough points for the buffer delay plus one epoch
            buffer_delay = buffer_id * self.points_per_step
            required_points = buffer_delay + self.points_per_epoch
            
        # Check if we have enough data points
        if len(self.all_previous_buffers_data[0]) < required_points:
            return False, f"Not enough data points (have {len(self.all_previous_buffers_data[0])}, need {required_points})"
            
        # For the first epoch, we can process it immediately
        if self.last_processed_buffer == -1:
            return True, ""
            
        # Get the current timestamp from the data
        current_timestamp = self.all_previous_buffers_data[self.buffer_timestamp_index][-1]
        
        # Get the last processed epoch's timestamp
        try:
            last_epoch_timestamp = self.all_previous_buffers_data[self.buffer_timestamp_index][
                self.processed_epoch_start_indices[self.last_processed_buffer][-1]
            ]
        except (IndexError, KeyError):
            # If we can't get the last epoch timestamp, we can still process
            return True, ""
            
        # Check if we've moved forward by at least buffer_step seconds
        time_since_last_epoch = current_timestamp - last_epoch_timestamp
        if time_since_last_epoch < self.buffer_step:
            return False, f"Not enough time has passed since last epoch ({time_since_last_epoch:.2f}s < {self.buffer_step}s)"
            
        return True, ""

    def next_available_epoch_on_buffer(self, buffer_id):
        """Return the next available epoch on the buffer"""
        
        # Check if we can process this buffer
        can_process, reason = self._has_enough_data_for_buffer(buffer_id)
        if not can_process:
            return False, reason, None, None
        

        epoch_start_idx, epoch_end_idx = self._get_next_epoch_indices(buffer_id)

        enough_data_to_process_epoch = self._enough_data_to_process_epoch(buffer_id, epoch_end_idx)
        # Check if we have enough data to process the epoch
        if not enough_data_to_process_epoch:
            return False, "Not enough data to process the epoch", None, None
        
        return True, None, epoch_start_idx, epoch_end_idx

    def manage_epoch(self, buffer_id, epoch_start_idx, epoch_end_idx):
        """Validate and process a specified epoch on a specified buffer."""


        # validate that we can process the epoch
        has_gap, gap_size = self.validate_epoch_gaps(buffer_id, epoch_start_idx, epoch_end_idx)

        if has_gap:
            # Handle the gap
            self.handle_gap(
                prev_timestamp=self.all_previous_buffers_data[self.buffer_timestamp_index][epoch_start_idx-1],
                gap_size=gap_size, buffer_id=buffer_id
            )
       
        # Update buffer status
        self.processed_epoch_start_indices[buffer_id].append(epoch_start_idx)
        self.last_processed_buffer = buffer_id  

        if has_gap:
            return

        # Process the epoch
        self._process_epoch(start_idx=epoch_start_idx, end_idx=epoch_end_idx, buffer_id=buffer_id)

    def _process_epoch(self, start_idx, end_idx, buffer_id):
        """Handle the data for a specified epoch on a specified buffer which has valid data."""        
        
        print(f"\nProcessing buffer {buffer_id}")
        print(f"Epoch range: {start_idx} to {end_idx}")
        print(f"Buffer {buffer_id}: Epoch range: {start_idx * self.expected_interval} to {end_idx * self.expected_interval} seconds")
        
        # Extract EXACTLY points_per_epoch data points from the correct slice
        epoch_data = np.array([
            self.all_previous_buffers_data[channel][start_idx:end_idx]
            for channel in self.electrode_channels
        ])
        
        # Verify we have exactly the right number of points
        assert epoch_data.shape[1] == self.points_per_epoch, f"Expected {self.points_per_epoch} points, got {epoch_data.shape[1]}"
        
        # Get the timestamp data for this epoch
        timestamp_data = self.all_previous_buffers_data[self.buffer_timestamp_index][start_idx:end_idx]
        epoch_start_time = timestamp_data[0]  # First timestamp in the epoch
        
        # Get sleep stage prediction using SignalProcessor
        sleep_stage, new_hidden_states = self.signal_processor.predict_sleep_stage(
            epoch_data,
            self.buffer_hidden_states[buffer_id]
        )
        
        print(f"Sleep stage: {self.visualizer.get_sleep_stage_text(sleep_stage[0])}")
        
        # Update visualization using Visualizer
        time_offset = start_idx / self.sampling_rate
        self.visualizer.plot_polysomnograph(
            epoch_data, 
            self.sampling_rate, 
            sleep_stage[0], 
            time_offset, 
            epoch_start_time
        )
        
        self.buffer_hidden_states[buffer_id] = new_hidden_states

    def save_to_csv(self, output_path):
        """Save raw data to CSV file"""
        self.output_csv_path = output_path
        if not self.saved_data:
            print("No data to save")
            return False
            
        try:
            # Convert to numpy array
            data_array = np.array(self.saved_data)
            
            # Create format specifiers to match original file
            fmt = ['%.6f'] * data_array.shape[1]  # Default to float format
            fmt[1] = '%d'  # Second column should be integer
            fmt[2:17] = ['%d'] * 15  # EEG channels should be integers
            
            # Save with exact format matching
            np.savetxt(output_path, data_array, delimiter='\t', fmt=fmt)
            print(f"Data saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            return False

    def validate_saved_csv(self, original_csv_path):
        """Validate that the saved CSV matches the original format exactly"""
        try:
            # Read both CSVs as strings first
            with open(self.output_csv_path, 'r') as f:
                saved_lines = f.readlines()
            with open(original_csv_path, 'r') as f:
                original_lines = f.readlines()
            
            print("\nCSV Validation Results:")
            
            # Check number of lines
            if len(saved_lines) != len(original_lines):
                print(f"❌ Line count mismatch: Original={len(original_lines)}, Saved={len(saved_lines)}")
                return False
            print(f"✅ Line count matches: {len(original_lines)} lines")
            
            # Compare each line exactly
            for i, (saved_line, original_line) in enumerate(zip(saved_lines, original_lines)):
                if saved_line != original_line:
                    print(f"❌ Line {i+1} does not match exactly:")
                    print(f"Original: {original_line.strip()}")
                    print(f"Saved:    {saved_line.strip()}")
                    return False
            
            print("✅ All lines match exactly")
            return True
        except Exception as e:
            print(f"Error validating CSV: {str(e)}")
            return False

    def _get_affected_buffer(self, timestamp):
        """
        Determine which buffer is affected by a gap based on timestamp.
        
        Args:
            timestamp: The timestamp where the gap occurred
            
        Returns:
            int: Buffer ID (0-5) or None if cannot be determined
        """
        if not self.all_previous_buffers_data[0]:  # No data yet
            return None
        
        # Calculate time from start of recording
        start_time = self.all_previous_buffers_data[self.buffer_timestamp_index][0]
        relative_time = timestamp - start_time
        
        # Calculate which buffer this timestamp would belong to
        buffer_id = int((relative_time % self.seconds_per_epoch) // 5)
        return buffer_id if 0 <= buffer_id <= 5 else None

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
        has_gap = False
        max_gap = 0
        gap_start_idx = None
        gap_end_idx = None

        # Check gap between chunks if we have a previous timestamp
        if prev_timestamp is not None:
            between_chunks_gap = timestamps[0] - prev_timestamp - self.expected_interval
            if abs(between_chunks_gap) >= self.gap_threshold:
                has_gap = True
                max_gap = between_chunks_gap
                gap_start_idx = -1  # -1 indicates gap is between chunks
                gap_end_idx = 0

        # Check gaps within the chunk
        for i in range(1, len(timestamps)):
            interval_deviation = timestamps[i] - timestamps[i-1] - self.expected_interval
            if abs(interval_deviation) >= self.gap_threshold:
                has_gap = True
                if abs(interval_deviation) > abs(max_gap):
                    max_gap = interval_deviation
                    gap_start_idx = i-1
                    gap_end_idx = i

        return has_gap, max_gap, gap_start_idx, gap_end_idx

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

    def _get_next_epoch_indices(self, buffer_id):
        """Get the start and end indices for a buffers next epoch based on the last processed epoch plus the points per epoch.
        
        Args:
            buffer_id: The ID of the buffer to get indices for
            
        Returns:
            tuple: (epoch_start_idx, epoch_end_idx)
            - epoch_start_idx: The starting index for the epoch
            - epoch_end_idx: The ending index for the epoch
        """
        buffer_start_offset = buffer_id * self.points_per_step
        last_epoch_start_idx = None
        
        try:
            last_epoch_start_idx = self.processed_epoch_start_indices[buffer_id][-1]
        except (IndexError, KeyError):
            pass

        # If this is the first epoch for this buffer, start at the buffer's offset
        if last_epoch_start_idx is None:
            epoch_start_idx = buffer_start_offset
        else:
            # Otherwise, start after the last processed epoch
            epoch_start_idx = last_epoch_start_idx + self.points_per_epoch

        epoch_end_idx = epoch_start_idx + self.points_per_epoch
        
        return epoch_start_idx, epoch_end_idx

    def _calculate_next_buffer_id_to_process(self):
        """Calculate the ID of the next buffer to process"""
        return (self.last_processed_buffer + 1) % 6

def update_status_line(message):
    """Update a single line of output, overwriting the previous line"""
    print(f"\r{message}", end="", flush=True)

def main():
    print(f"Ganglion timestamp channel: {BoardShim.get_timestamp_channel(BoardIds.GANGLION_BOARD)}")
    print(f"Cyton timestamp channel: {BoardShim.get_timestamp_channel(BoardIds.CYTON_BOARD)}")
    print(f"Ganglion EXG channels: {BoardShim.get_exg_channels(BoardIds.GANGLION_BOARD)}")
    print(f"Cyton EXG channels: {BoardShim.get_exg_channels(BoardIds.CYTON_BOARD)}")

    # Create data acquisition instance
    # input_file = "data/cyton_BrainFlow-gap_short.csv"
    # input_file = "gssc/sandbox/test_data"
    # input_file = "data/tiny_gap.csv"
    # input_file = "data/cyton_BrainFlow-adjusted-timestamps.csv"
    # input_file = "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.csv"   
    # input_file = "test_data/gapped_data.csv"
    
    input_file = "data/test_data/consecutive_data.csv"
    data_acquisition = DataAcquisition(input_file)
    
    # Set up output file path
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"processed_{timestamp}.csv")
    
    try:
        # Setup and start board
        board_shim = data_acquisition.setup_board()
        print("\nBoard setup complete")
        
        initial_count = data_acquisition.start_stream()
        print("\nStream started successfully")
        
        # Create buffer manager and set recording start time
        buffer_manager = BufferManager(board_shim, data_acquisition.sampling_rate)
        data_acquisition.set_buffer_manager(buffer_manager)
        buffer_manager.visualizer.recording_start_time = data_acquisition.recording_start_time
        
        # Wait a short time for initial data
        print("\nWaiting for initial data...")
        time.sleep(0.5)
        
        # Get initial data
        initial_data = data_acquisition.get_initial_data()
        if initial_data.size > 0:
            success = buffer_manager.add_data(initial_data, is_initial=True)
            print(f"\nInitial data processing:")
            print(f"- Added to buffer: {'Success' if success else 'Failed'}")
            print(f"- Samples: {len(initial_data[0])}")
        else:
            print("\nWarning: No initial data available")
        
        consecutive_empty_count = 0
        sleep_time = 0.1
        iteration_count = 0
        
        print("\nEntering main processing loop...")
        
        while True:
            iteration_count += 1
            if DEBUG_VERBOSE:
                print(f"\nIteration {iteration_count}:")
            
            # Get new data
            new_data = data_acquisition.get_new_data()
            
            # Log timestamps if we have data
            if new_data.size > 0 and data_acquisition.timestamp_channel is not None:
                start_timestamp = new_data[data_acquisition.timestamp_channel][0]
                end_timestamp = new_data[data_acquisition.timestamp_channel][-1]
                
                # Set recording start time if not set
                if data_acquisition.recording_start_time is None:
                    data_acquisition.recording_start_time = start_timestamp
                
                # Calculate total duration from start
                total_duration = end_timestamp - data_acquisition.recording_start_time
                hours = int(total_duration // 3600)
                minutes = int((total_duration % 3600) // 60)
                seconds = total_duration % 60
                
                # Calculate chunk duration
                chunk_duration = end_timestamp - start_timestamp
                
                print(f"\r\033[2KTimestamps: Start={start_timestamp:.3f}s, End={end_timestamp:.3f}s, Chunk={chunk_duration:.3f}s, Total={hours}h {minutes}m {seconds:.3f}s", end="", flush=True)
            
            # Handle empty or invalid data
            if new_data.size == 0:
                consecutive_empty_count += 1
                sleep_time = min(1.0, sleep_time * 1.5)
                print("\r\033[2K", end="", flush=True)  # Clear the entire line
                time.sleep(sleep_time)
                continue
            
            # Successfully got data
            if not buffer_manager.add_data(new_data):
                print("\r\033[2K", end="", flush=True)  # Clear the entire line
                print("\nFailed to add new data to buffer:")
                print(f"- Data shape: {new_data.shape}")
                continue
            
            consecutive_empty_count = 0
            sleep_time = 0.1
            next_buffer_id = buffer_manager._calculate_next_buffer_id_to_process()
            print(f"\nAttempting to process buffer {next_buffer_id}")
            print(f"Total data points collected: {buffer_manager.points_collected}")
            print(f"Last processed buffer: {buffer_manager.last_processed_buffer}")

            # Process next epoch on next buffer
            can_process, reason, epoch_start_idx, epoch_end_idx = buffer_manager.next_available_epoch_on_buffer(next_buffer_id)

            if can_process:
                # Process the buffer
                print("\r\033[2K", end="", flush=True)  # Clear the entire line
                buffer_manager.manage_epoch(buffer_id=next_buffer_id, epoch_start_idx=epoch_start_idx, epoch_end_idx=epoch_end_idx)
            else:
                print(f"Cannot process buffer {next_buffer_id}: {reason}")

            
            # Check for end of file
            if (data_acquisition.file_data is not None and 
                data_acquisition.last_chunk_last_timestamp is not None and 
                data_acquisition.last_chunk_last_timestamp >= data_acquisition.file_data.iloc[-1, data_acquisition.timestamp_channel]):
                print()  # Add newline to clean up the last timestamp log
                print(f"\nReached end of file:")
                print(f"- Final timestamp: {data_acquisition.last_chunk_last_timestamp}")
                print(f"- Total iterations: {iteration_count}")
                
                # Save processed data to CSV
                print("\nSaving processed data to CSV...")
                if buffer_manager.save_to_csv(output_file):
                    print(f"Data saved to {output_file}")
                    
                    # Validate saved CSV
                    print("\nValidating saved CSV...")
                    if buffer_manager.validate_saved_csv(input_file):
                        print("✅ CSV validation passed!")
                    else:
                        print("❌ CSV validation failed!")
                break
                
            time.sleep(0.1)
            
    except BaseException as e:
        print()  # Add newline to clean up the last timestamp log
        logging.warning('Exception occurred:', exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        print()  # Add newline to clean up the last timestamp log
        print("\nCleaning up...")
        data_acquisition.release()
        print("Session ended")


if __name__ == '__main__':
    main()

