import os
import time
import threading
import logging
import numpy as np
import pandas as pd
from typing import Optional

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from ..signal.buffer import BufferManager

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
        self._streaming = False  # Add streaming state flag
    
    def set_buffer_manager(self, buffer_manager):
        self.buffer_manager = buffer_manager
    
    def setup_board(self):
        """Initialize and setup the board for data collection"""
        try:
            print("\rStarting board setup...")
            
            # Set environment variable to unblock threads if operations hang
            os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '2'
            
            print("\rInitializing board logger...")
            # Run enable_dev_board_logger in a separate thread with timeout
            logger_thread = threading.Thread(target=BoardShim.enable_dev_board_logger)
            logger_thread.daemon = True
            logger_thread.start()
            logger_thread.join(timeout=2.0)  # Wait up to 2 seconds
            
            if logger_thread.is_alive():
                print("\rWarning: BoardShim.enable_dev_board_logger() is taking too long, continuing setup...")
            
            print("\rInitializing board parameters...")
            params = BrainFlowInputParams()
            params.board_id = BoardIds.PLAYBACK_FILE_BOARD
            params.master_board = self.master_board_id
            params.file = self.file_path
            params.playback_file_max_count = 1
            params.playback_speed = 1
            params.playback_file_offset = 0

            print("\rBoard Parameters Configuration:")
            print(f"\r- Board ID: {params.board_id} (PLAYBACK_FILE_BOARD)")
            print(f"\r- Master Board: {params.master_board} (CYTON_DAISY_BOARD)")
            print(f"\r- File Path: {params.file}")
            print(f"\r- Playback File Max Count: {params.playback_file_max_count}")
            print(f"\r- Playback Speed: {params.playback_speed}")
            print(f"\r- Playback File Offset: {params.playback_file_offset}")
            
            print(f"\rCreating board with parameters...")
            self.board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
            print("\rBoard created successfully")
            
            print("\rPreparing board session...")
            # Run prepare_session in a separate thread with timeout
            prepare_thread = threading.Thread(target=self.board_shim.prepare_session)
            prepare_thread.daemon = True
            prepare_thread.start()
            prepare_thread.join(timeout=2.0)  # Wait up to 2 seconds
            
            if prepare_thread.is_alive():
                print("\rWarning: prepare_session() is taking too long, forcing cleanup...")
                # If we get here, prepare_session is stuck
                self.board_shim = None
                raise RuntimeError("prepare_session() timed out")
            
            print("\rBoard session prepared successfully")
            
            # Get sampling rate and timestamp channel from master board, not playback board
            print("\rGetting board configuration...")
            self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.master_board_id)
            
            print("\rBoard Configuration:")
            print(f"\r- Master board ID: {self.master_board_id}")
            print(f"\r- Timestamp channel: {self.timestamp_channel}")
            print(f"\r- Sampling rate: {self.sampling_rate}")
            
            print("\rConfiguring board for old timestamps...")
            self.board_shim.config_board("old_timestamps")
            print("\rBoard configured for old timestamps")

            # Load the entire file data at initialization
            print(f"\rLoading data from file: {self.file_path}")
            self.file_data = pd.read_csv(self.file_path, sep='\t', dtype=float)
            print(f"\rLoaded file with {len(self.file_data)} samples")
            
            print("\rBoard setup completed successfully")
            return self.board_shim
            
        except Exception as e:
            print(f"\rFailed to setup board: {str(e)}")
            raise

    def start_stream(self):
        """Start the data stream"""
        if not self.board_shim:
            raise RuntimeError("\rBoard not initialized. Call setup_board first.")
        
        if self._streaming:
            print("\rStream is already running")
            return
            
        try:
            print("\rStarting data stream...")
            print(f"\rBuffer size: {self.buffer_size}")
            print(f"\rSampling rate: {self.sampling_rate}")
            print(f"\rTimestamp channel: {self.timestamp_channel}")
            
            print("\rCalling board_shim.start_stream()...")
            self.board_shim.start_stream(self.buffer_size)
            self._streaming = True
            print("\rStream started successfully")
            
            # Initialize recording_start_time as None - we'll set it when we get the first data packet
            self.recording_start_time = None
            
            # Wait briefly for stream to start
            print("\rWaiting for stream to initialize...")
            time.sleep(0.1)
            
            initial_count = self.board_shim.get_board_data_count()
            print(f"\rInitial data count after stream start: {initial_count}")
            return initial_count
            
        except Exception as e:
            self._streaming = False
            print(f"\rFailed to start stream: {str(e)}")
            raise

    def stop_stream(self):
        """Stop the data stream"""
        if not self._streaming:
            print("\rStream is not running")
            return
            
        try:
            if self.board_shim and self.board_shim.is_prepared():
                # Check if we're in a gap by looking at stream processor's consecutive empty count
                # We can access it through buffer_manager since it has a reference to stream_processor
                in_gap = (hasattr(self.buffer_manager, 'stream_processor') and 
                         self.buffer_manager.stream_processor and 
                         self.buffer_manager.stream_processor.consecutive_empty_count > 0)
                
                if in_gap:
                    print("\rDetected gap, forcing session release...")
                    try:
                        # Set environment variable to unblock threads if release hangs
                        os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '1'
                        # Try release with a timeout using threading
                        release_thread = threading.Thread(target=self.board_shim.release_session)
                        release_thread.daemon = True
                        release_thread.start()
                        release_thread.join(timeout=2.0)  # Wait up to 2 seconds
                        
                        if release_thread.is_alive():
                            print("\rRelease session taking too long, forcing cleanup...")
                            # If we get here, release_session is stuck
                            # Force cleanup by nullifying the board
                            self.board_shim = None
                            self._streaming = False
                        else:
                            # Release completed successfully
                            print("\rSession released successfully")
                            # Prepare new session
                            self.setup_board()
                    except Exception as e:
                        print(f"\rError during forced release: {str(e)}")
                        self.board_shim = None
                        self._streaming = False
                else:
                    # Normal clean stop if we're not in a gap
                    self.board_shim.stop_stream()
                    self._streaming = False
                    print("\rStream stopped successfully")
                
        except Exception as e:
            print(f"\rFailed to stop stream: {str(e)}")
            # Force cleanup on any error
            self.board_shim = None
            self._streaming = False
            raise

    def is_streaming(self):
        """Check if the stream is currently running"""
        return self._streaming

    def get_initial_data(self):
        """Get initial data from the board"""
        if not self.board_shim:
            raise RuntimeError("Board not initialized. Call setup_board first.")
        
        try:
            # Log data count before getting data
            data_count = self.board_shim.get_board_data_count()
            print(f"\rAttempting to get initial data. Available data count: {data_count}")
            
            if data_count > 0:
                print(f"\rGetting {data_count} samples of initial data...")
                # Get all available data
                initial_data = self.board_shim.get_board_data(data_count)

                # Log details about the retrieved data
                if initial_data.size <= 0:
                    print("\rget_board_data returned empty array despite positive count!")
                else:
                    print(f"\rRetrieved initial data shape: {initial_data.shape}")
                    # For streaming, set recording_start_time to the first timestamp if not already set
                    if self.recording_start_time is None and self.timestamp_channel is not None:
                        self.recording_start_time = initial_data[self.timestamp_channel][0]
                        print(f"\rSet recording start time to: {self.recording_start_time}")
                    
                return initial_data
            else:
                print("\rNo initial data available (data count is 0)")
                return np.array([])
            
        except Exception as e:
            print(f"\rFailed to get initial data: {str(e)}")
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
        if not self._streaming:
            raise RuntimeError("Cannot get data: stream is not running")
            
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
                    print("\rget_board_data returned empty array despite positive count!")
                    return np.array([])
            else:
                return np.array([])
            
        except Exception as e:
            print(f"\rFailed to get new data: {str(e)}")
            raise

    def release(self):
        """Release the board session"""
        if self.board_shim and self.board_shim.is_prepared():
            try:
                if self._streaming:
                    self.stop_stream()
                self.board_shim.release_session()
                self._streaming = False
                print("\rSession released successfully")
            except Exception as e:
                print(f"\rFailed to release session: {str(e)}")
                raise 