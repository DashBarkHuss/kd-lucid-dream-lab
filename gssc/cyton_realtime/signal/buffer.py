import numpy as np
import torch
import logging
import threading
from typing import List, Tuple, Optional
from montage import Montage
from brainflow.board_shim import BoardShim

from ..visualization.pyqt_visualizer import PyQtVisualizer
from .processor import SignalProcessor

class BufferManager:
    """Manages data buffers and their processing"""
    def __init__(self, board_shim, sampling_rate, montage: Montage = None):
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
        self.processed_epoch_start_indices = [[] for _ in range(6)]
        
        # Initialize hidden states for each buffer
        self.buffer_hidden_states = [
            [torch.zeros(10, 1, 256) for _ in range(7)]  # 7 hidden states for 7 combinations
            for _ in range(6)  # 6 buffers (0s to 25s in 5s steps)
        ]
        self.signal_processor = SignalProcessor()
        self.visualizer = PyQtVisualizer(self.seconds_per_epoch, self.board_shim, montage)
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
        self.shutdown_event = threading.Event()  # For signaling threads to stop
        self.threads_stopped = threading.Event()  # For threads to signal they've stopped
        self.active_threads = 0  # Counter for active threads
        self.thread_lock = threading.Lock()  # For thread counter synchronization

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
            return False, f"\rChannel index {adjusted_channel_idx} out of range"
            
        channel_data = data[adjusted_channel_idx]
        
        # For the first validation, just store the last value
        if self.last_validated_value is None:
            self.last_validated_value = channel_data[-1]
            return True, "First validation - stored last value"
            
        # Check if the new data starts where the last data ended
        expected_next_value = self.last_validated_value + 1
        if channel_data[0] != expected_next_value:
            return False, f"\rNon-consecutive data detected. Expected {expected_next_value}, got {channel_data[0]}"
            
        # Check if all values in the chunk are consecutive
        for i in range(1, len(channel_data)):
            if channel_data[i] != channel_data[i-1] + 1:
                return False, f"\rNon-consecutive data within chunk at index {i}. Expected {channel_data[i-1] + 1}, got {channel_data[i]}"
                
        # Update last validated value
        self.last_validated_value = channel_data[-1]
        return True, "Data validated successfully"

    def add_data(self, new_data, is_initial=False):
        """Add new data to the buffer"""
        # Validate data values
        if np.any(np.isnan(new_data)) or np.any(np.isinf(new_data)):
            logging.warning("\rData contains NaN or infinite values!")
            return False
            
        # Validate consecutive values if enabled
        if self.validate_consecutive_values:
            is_valid, message = self.validate_consecutive_data(new_data)
            if not is_valid:
                #  throw an exception
                raise Exception(f"\rConsecutive value validation failed: {message}")
            
        # Update points collected
        self.points_collected += len(new_data[0])
        
        # Update all_previous_data
        if not is_initial:
            for i, channel in enumerate(self.all_channels_with_timestamp):
                self.all_previous_buffers_data[i].extend(new_data[channel].tolist())
        else:
            for i, channel in enumerate(self.all_channels_with_timestamp):
                self.all_previous_buffers_data[i] = new_data[channel].tolist()
                
        return True

    def save_new_data(self, new_data, is_initial=False):
        """Save new data to the saved_data buffer for later CSV export"""
        if not hasattr(self, 'saved_data'):
            self.saved_data = []
            
        new_rows = new_data.T.tolist()
        
        # For initial data, save everything
        if is_initial:
            self.saved_data.extend(new_rows)
            if new_rows:
                self.last_saved_timestamp = new_rows[-1][self.buffer_timestamp_index]
            return
            
        # For subsequent data, only filter out exact duplicates
        if self.last_saved_timestamp is not None:
            # Find the first row with a timestamp greater than the last saved timestamp
            start_idx = 0
            for i, row in enumerate(new_rows):
                if row[self.buffer_timestamp_index] > self.last_saved_timestamp:
                    start_idx = i
                    break
            
            # Save all rows from that point forward
            if start_idx < len(new_rows):
                self.saved_data.extend(new_rows[start_idx:])
                self.last_saved_timestamp = new_rows[-1][self.buffer_timestamp_index]
        else:
            # If no last saved timestamp, save all rows
            self.saved_data.extend(new_rows)
            if new_rows:
                self.last_saved_timestamp = new_rows[-1][self.buffer_timestamp_index]

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
        # Check if we have enough data points
        if buffer_id == 0:
            required_points = self.points_per_epoch
        else:
            buffer_delay = buffer_id * self.points_per_step
            required_points = buffer_delay + self.points_per_epoch
            
        if len(self.all_previous_buffers_data[0]) < required_points:
            return False, "Not enough data points"
            
        # Get the current timestamp from the data
        current_timestamp = self.all_previous_buffers_data[self.buffer_timestamp_index][-1]
        
        # For the first epoch, we can process it
        if self.last_processed_buffer == -1:
            return True, ""
            
        # For subsequent epochs, check if we've moved forward by buffer_step seconds
        last_epoch_timestamp = self.all_previous_buffers_data[self.buffer_timestamp_index][
            self.processed_epoch_start_indices[self.last_processed_buffer][-1]
        ]
        
        if current_timestamp - last_epoch_timestamp < self.buffer_step:
            return False, "Not enough time has passed"
            
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
        
        print(f"\rProcessing buffer {buffer_id}")
        print(f"\rEpoch range: {start_idx} to {end_idx}")
        print(f"\rBuffer {buffer_id}: Epoch range: {start_idx * self.expected_interval} to {end_idx * self.expected_interval} seconds")
        
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
        
        print(f"\rSleep stage: {self.visualizer.get_sleep_stage_text(sleep_stage[0])}")
        
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
            print("\rNo data to save")
            return False
            
        try:
            # Convert to numpy array
            data_array = np.array(self.saved_data)
            
            # Create format specifiers to match original file
            # All columns should use %.6f for consistent floating-point precision
            fmt = ['%.6f'] * data_array.shape[1]
            
            # Save with exact format matching
            np.savetxt(output_path, data_array, delimiter='\t', fmt=fmt)
            print(f"\rData saved to {output_path}")
            return True
        except Exception as e:
            print(f"\rError saving to CSV: {str(e)}")
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
                print(f"\r❌ Line count mismatch: Original={len(original_lines)}, Saved={len(saved_lines)}")
                return False
            print(f"\r✅ Line count matches: {len(original_lines)} lines")
            
            # Compare each line exactly
            for i, (saved_line, original_line) in enumerate(zip(saved_lines, original_lines)):
                if saved_line != original_line:
                    print(f"\r❌ Line {i+1} does not match exactly:")
                    print(f"\rOriginal: {original_line.strip()}")
                    print(f"\rSaved:    {saved_line.strip()}")
                    return False
            
            print("\r✅ All lines match exactly")
            return True
        except Exception as e:
            print(f"\rError validating CSV: {str(e)}")
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
            
            print(f"\rBuffer {buffer_id} reset complete - hidden states cleared")

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