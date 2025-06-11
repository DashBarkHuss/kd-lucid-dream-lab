from gssc_local.realtime_with_restart.data_manager import DataManager
from gssc_local.realtime_with_restart.board_manager import BoardManager
from logging import Logger
from gssc_local.montage import Montage

class ReceivedStreamedDataHandler: 
    """Handles processing and storage of incoming EEG data"""
    def __init__(self, board_manager: BoardManager, logger: Logger):
        self.sample_count = 0  # Total number of samples processed
        self.board_manager = board_manager
        self.data_manager = DataManager(self.board_manager.board_shim, self.board_manager.sampling_rate, Montage.minimal_sleep_montage())
        self.logger = logger
        self.is_initial = True

    def process_board_data(self, board_data):
        """Process incoming data chunk"""
        self.logger.info("\n=== ReceivedStreamedDataHandler.process_board_data ===")
        self.logger.info(f"Received data shape: {board_data.shape}")
        
        self.sample_count += board_data.shape[1]  # Increment total sample count

        # Queue data for CSV writing
        self.data_manager.queue_data_for_csv_write(board_data, self.is_initial)
        # add data to memory for epoch processing
        self.data_manager.add_to_data_processing_buffer(board_data, self.is_initial)

        self.is_initial = False
        
        # Calculate which buffer should be processed next
        next_buffer_id = self.data_manager._calculate_next_buffer_id_to_process()
        self.logger.info(f"Next buffer ID to process: {next_buffer_id}")

        # Check if we CAN process an epoch (have enough data + right timing)
        can_process, reason, epoch_start_idx, epoch_end_idx = self.data_manager.next_available_epoch_on_buffer(next_buffer_id)
        self.logger.info(f"Can process epoch: {can_process}")
        if not can_process:
            self.logger.info(f"Reason: {reason}")

        if can_process:
            # Only process when we have a complete epoch ready
            self.logger.info(f"Processing epoch on buffer {next_buffer_id}")
            self.logger.info(f"Epoch indices: {epoch_start_idx} to {epoch_end_idx}")
            self.data_manager.manage_epoch(buffer_id=next_buffer_id, 
                                    epoch_start_idx=epoch_start_idx, 
                                    epoch_end_idx=epoch_end_idx)
        
        # Log processing statistics
        self.logger.info(f"Processed {self.sample_count} samples")        
        # Calculate and log basic statistics for monitoring
        