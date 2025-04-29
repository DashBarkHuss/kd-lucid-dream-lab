import time
import threading
import numpy as np
import logging
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone, timedelta

from ..config import DEBUG_VERBOSE
from ..signal.buffer import BufferManager
from ..utils.time import format_timestamp_to_hawaii, format_elapsed_time

if TYPE_CHECKING:
    from ..acquisition.board import DataAcquisition

class DataStreamProcessor:
    """Handles the main processing loop and stream control"""
    def __init__(self, data_acquisition: 'DataAcquisition', buffer_manager: BufferManager):
        self.data_acquisition = data_acquisition
        self.buffer_manager = buffer_manager
        self._processing = False
        self._processing_thread = None
        self.consecutive_empty_count = 0
        self.sleep_time = 0.1
        self.iteration_count = 0
        self._timestamp_thread = None
        self._timestamp_running = False
        self.output_lock = threading.Lock()
        self._new_data_received = False  # Flag to track new data
        
    def _timestamp_loop(self):
        """Thread for handling timestamp output"""
        while self._timestamp_running:
            try:
                if self.data_acquisition.last_chunk_last_timestamp is not None and self._new_data_received:
                    total_duration = self.data_acquisition.last_chunk_last_timestamp - self.data_acquisition.recording_start_time
                    start_time = format_timestamp_to_hawaii(self.data_acquisition.recording_start_time)
                    current_time = format_timestamp_to_hawaii(self.data_acquisition.last_chunk_last_timestamp)
                    elapsed_time = format_elapsed_time(total_duration)
                    
                    with self.output_lock:
                        print(f"\rTimestamps: Start={start_time} | Current={current_time} | Elapsed={elapsed_time}", end='', flush=True)
                    self._new_data_received = False  # Reset the flag after showing timestamp
                time.sleep(0.1)
            except Exception as e:
                print(f"\rTimestamp thread error: {str(e)}")
                
    def _process_loop(self):
        """Main processing loop"""
        while self._processing:
            self.iteration_count += 1
            if DEBUG_VERBOSE:
                print(f"\rIteration {self.iteration_count}")
            
            try:
                # Get new data
                new_data = self.data_acquisition.get_new_data()
                
                # Handle empty or invalid data
                if new_data.size == 0:
                    self.consecutive_empty_count += 1
                    self.sleep_time = min(1.0, self.sleep_time * 1.5)
                    with self.output_lock:
                        print(f"\rNo data received. Sleeping for {self.sleep_time:.2f}s. Empty count: {self.consecutive_empty_count}")
                    time.sleep(self.sleep_time)
                    continue
                
                # Successfully got data
                if self.buffer_manager.add_data(new_data):
                    # Save data first, before any processing
                    self.buffer_manager.save_new_data(new_data)
                    self._new_data_received = True  # Set flag when new data is received
                    
                    # Then try to process
                    self.consecutive_empty_count = 0
                    self.sleep_time = 0.1
                    next_buffer_id = self.buffer_manager._calculate_next_buffer_id_to_process()

                    # Process next epoch on next buffer
                    can_process, reason, epoch_start_idx, epoch_end_idx = self.buffer_manager.next_available_epoch_on_buffer(next_buffer_id)

                    if can_process:
                        self.buffer_manager.manage_epoch(buffer_id=next_buffer_id, epoch_start_idx=epoch_start_idx, epoch_end_idx=epoch_end_idx)
                else:
                    print(f"\rFailed to add new data to buffer. Data shape: {new_data.shape}")
                    # Still try to save the data even if processing failed
                    self.buffer_manager.save_new_data(new_data)
                    continue
                
                # Check for end of file
                if (self.data_acquisition.file_data is not None and 
                    self.data_acquisition.last_chunk_last_timestamp is not None and 
                    self.data_acquisition.last_chunk_last_timestamp >= self.data_acquisition.file_data.iloc[-1, self.data_acquisition.timestamp_channel]):
                    print(f"\rReached end of file. Final timestamp: {self.data_acquisition.last_chunk_last_timestamp}. Total iterations: {self.iteration_count}")
                    self._processing = False
                    break
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\rError in processing loop: {str(e)}")
                self._processing = False
                raise
                
    def start_processing(self):
        """Start the processing loop"""
        if self._processing:
            print("\rProcessing is already running")
            return
            
        self._processing = True
        self._timestamp_running = True
        # Start processing in a separate thread
        self._processing_thread = threading.Thread(target=self._process_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        # Start timestamp thread
        self._timestamp_thread = threading.Thread(target=self._timestamp_loop)
        self._timestamp_thread.daemon = True
        self._timestamp_thread.start()
        
    def stop_processing(self):
        """Stop the processing loop"""
        if not self._processing:
            print("\rProcessing is not running")
            return
            
        # Signal the threads to stop
        self._processing = False
        self._timestamp_running = False
        
        # Only try to join if we're not in the processing thread
        if (self._processing_thread and 
            self._processing_thread.is_alive() and 
            threading.current_thread() != self._processing_thread):
            self._processing_thread.join(timeout=1.0)
            
        if (self._timestamp_thread and 
            self._timestamp_thread.is_alive() and 
            threading.current_thread() != self._timestamp_thread):
            self._timestamp_thread.join(timeout=1.0)
        
        self._processing_thread = None
        self._timestamp_thread = None 