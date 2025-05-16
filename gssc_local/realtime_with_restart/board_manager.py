# previously called DataAcquisition
import time
import pandas as pd
import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class BoardManager:
    """Handles board setup and data collection"""
    def __init__(self, file_path: str = None, master_board_id: BoardIds = BoardIds.CYTON_DAISY_BOARD):
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
        self.master_board_id = master_board_id  # Add master board ID
        self.timestamp_channel = None  # Will be set in setup_board
        self.gap_threshold = 2.0  # Add gap threshold in seconds
        

    def set_buffer_manager(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def set_board_shim(self):
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

        params = BrainFlowInputParams()
        params.board_id = BoardIds.PLAYBACK_FILE_BOARD
        params.master_board = self.master_board_id  # Use class variable
        # params.file = self.file_path
        params.playback_file_max_count = 1
        params.playback_speed = 1
        params.playback_file_offset = 0

        self.board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

        # Get sampling rate and timestamp channel from master board, not playback board
        self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.master_board_id)
        
        return self.board_shim

    def setup_board(self):
        """Initialize and setup the board for data collection"""
        self.set_board_shim()
        
        try:
            self.board_shim.prepare_session()
            print(f"\nBoard Configuration:")
            print(f"Master board: {self.master_board_id}")
            print(f"Timestamp channel: {self.timestamp_channel}")
            print(f"Sampling rate: {self.sampling_rate}")
            
            self.board_shim.config_board("old_timestamps")

            # Load the entire file data at initialization
            self.file_data = pd.read_csv(self.file_path, sep='\t', header=None, dtype=float)
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







