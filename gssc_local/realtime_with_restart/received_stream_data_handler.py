from data_manager import DataManager
from board_manager import BoardManager
from logging import Logger
from gssc_local.montage import Montage

class ReceivedStreamedDataHandler: 
    """Handles processing and storage of incoming EEG data"""
    def __init__(self, board_manager: BoardManager, logger: Logger):
        # Initialize buffers to store data chunks and their timestamps
        self.data_buffer = []  # Stores raw EEG data chunks
        self.timestamp_buffer = []  # Stores corresponding timestamps
        self.sample_count = 0  # Total number of samples processed
        self.board_manager = board_manager
        self.data_manager = DataManager(self.board_manager.board_shim, self.board_manager.sampling_rate, Montage.minimal_sleep_montage())
        self.logger = logger
    def process_board_data(self, board_data):
        """Process incoming data chunk and store it"""
        # Store the new data chunk and its timestamps
        self.data_buffer.append(board_data)
        timestamps = board_data[self.board_manager.timestamp_channel]
        self.timestamp_buffer.append(timestamps)

        self.sample_count += board_data.shape[1]  # Increment total sample count

        # Successfully got data
        if self.data_manager.add_data(board_data): # add data to the full buffer. this only saves the relevant data to the buffer and it validates the data
            # First just save the raw data
            self.data_manager.save_new_data(board_data)
            
            # Calculate which buffer should be processed next
            next_buffer_id = self.data_manager._calculate_next_buffer_id_to_process()

            # Check if we CAN process an epoch (have enough data + right timing)
            can_process, reason, epoch_start_idx, epoch_end_idx = self.data_manager.next_available_epoch_on_buffer(next_buffer_id)

            if can_process:
                # Only process when we have a complete epoch ready
                sleep_stage = self.data_manager.manage_epoch(buffer_id=next_buffer_id, 
                                        epoch_start_idx=epoch_start_idx, 
                                        epoch_end_idx=epoch_end_idx)
                # add the sleep stage to the csv at the epoch end idx and include the buffer id
                self.data_manager.add_sleep_stage_to_csv(sleep_stage, next_buffer_id, epoch_end_idx)
        
        # Log processing statistics
        self.logger.info(f"Processed {self.sample_count} samples")        
        # Calculate and log basic statistics for monitoring
        