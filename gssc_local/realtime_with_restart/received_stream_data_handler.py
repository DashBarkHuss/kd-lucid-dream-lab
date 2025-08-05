from gssc_local.realtime_with_restart.data_manager import DataManager
from gssc_local.realtime_with_restart.board_manager import BoardManager
from logging import Logger
from gssc_local.montage import Montage
from gssc_local.realtime_with_restart.utils.data_filtering_utils import sanitize_data










class ReceivedStreamedDataHandler: 
    """Handles processing and storage of incoming EEG data"""
    def __init__(self, board_manager: BoardManager, logger: Logger, montage: Montage = None):
        self.sample_count = 0  # Total number of samples processed
        self.board_manager = board_manager
        self.data_manager = DataManager(self.board_manager.board_shim, self.board_manager.sampling_rate, montage)
        self.logger = logger

    def process_board_data(self, board_data):
        """Process incoming data chunk"""
        self.logger.info("\n=== ReceivedStreamedDataHandler.process_board_data ===")
        self.logger.info(f"Received data shape: {board_data.shape}")
        

        # Get last saved timestamp for filtering
        last_saved_timestamp = self.data_manager.csv_manager.last_saved_timestamp
        is_initial = last_saved_timestamp is None
        
        # Sanitize raw data to remove duplicates and fix ordering issues
        filtered_data = sanitize_data(
            board_data, 
            self.board_manager.board_timestamp_channel, 
            self.logger,
            last_saved_timestamp=last_saved_timestamp,
            expected_sample_rate=self.board_manager.sampling_rate
        )
        
        
        self.sample_count += filtered_data.shape[1]  # Increment total sample count

        # Queue data for CSV writing
        self.data_manager.queue_data_for_csv_write(filtered_data, is_initial)
        # add data to memory for epoch processing
        self.data_manager.add_to_data_processing_buffer(filtered_data, is_initial)
        
        # Calculate which buffer should be processed next
        next_buffer_id = self.data_manager._calculate_next_buffer_id_to_process()
        self.logger.info(f"Next buffer ID to process: {next_buffer_id}")

        # Check if we CAN process an epoch (have enough data + right timing)
        can_process, reason, epoch_start_idx, epoch_end_idx = self.data_manager.next_available_epoch_on_round_robin_buffer(next_buffer_id)
        self.logger.info(f"Can process epoch: {can_process}")
        if not can_process:
            self.logger.info(f"Reason: {reason}")

        if can_process:
            # Only process when we have a complete epoch ready

            # Process the epoch
            self.data_manager.manage_epoch(buffer_id=next_buffer_id, 
                                    epoch_start_idx_abs=epoch_start_idx, 
                                    epoch_end_idx_abs=epoch_end_idx)
            

          
            epoch_start_idx_abs, _ = self.data_manager._get_next_epoch_indices(self.data_manager._calculate_next_buffer_id_to_process())
            # Only trim buffer after successful epoch processing
            # This ensures we don't trim data that hasn't been processed yet
            self.data_manager.etd_buffer_manager.trim_buffer(
                max_next_expected=epoch_start_idx_abs,
            )

        # Log processing statistics
        self.logger.info(f"Collected {self.sample_count} processed samples")


        # Calculate and log basic statistics for monitoring
        