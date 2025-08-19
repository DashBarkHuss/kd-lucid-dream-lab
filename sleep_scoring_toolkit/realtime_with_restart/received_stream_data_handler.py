from sleep_scoring_toolkit.realtime_with_restart.data_manager import DataManager
from sleep_scoring_toolkit.realtime_with_restart.board_manager import BoardManager
from logging import Logger
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.realtime_with_restart.utils.data_filtering_utils import sanitize_data
from sleep_scoring_toolkit.realtime_with_restart.utils.session_utils import generate_session_timestamp
import numpy as np










class ReceivedStreamedDataHandler: 
    """Handles processing and storage of incoming EEG data"""
    def __init__(self, board_manager: BoardManager, logger: Logger, montage: Montage = None, event_dispatcher=None, session_timestamp=None):
        self.sample_count = 0  # Total number of samples processed
        self.board_manager = board_manager
        
        # Generate session timestamp if not provided
        if session_timestamp is None:
            session_timestamp = generate_session_timestamp()
        self.session_timestamp = session_timestamp
        
        self.data_manager = DataManager(self.board_manager.board_shim, self.board_manager.sampling_rate, montage, event_dispatcher, session_timestamp)
        self.logger = logger

    def process_board_data_chunk(self, raw_board_data_chunk):
        """Process incoming data chunk"""
        self.logger.info("\n=== ReceivedStreamedDataHandler.process_board_data ===")
        self.logger.info(f"Received data shape: {raw_board_data_chunk.shape}")
        

        # Get last saved timestamp for filtering
        last_saved_timestamp = self.data_manager.csv_manager.last_saved_timestamp
        is_initial = last_saved_timestamp is None
        
        # Sanitize raw data to remove duplicates and fix ordering issues
        sanitized_board_data_chunk = sanitize_data(
            raw_board_data_chunk, 
            self.board_manager.board_timestamp_channel, 
            self.logger,
            last_saved_timestamp=last_saved_timestamp,
            expected_sample_rate=self.board_manager.sampling_rate
        )
        
        
        self.sample_count +=  sanitized_board_data_chunk.shape[1]  # Increment total sample count

        # Queue data for CSV writing (extract raw data for CSV manager)
        self.data_manager.queue_data_for_csv_write(sanitized_board_data_chunk.data, is_initial)
    
    
       
        # Validate data before adding to buffer
        if not self.data_manager.validate_data(sanitized_board_data_chunk):
            self.logger.error("Data validation failed, skipping epoch processing")
            return
            
        # add data directly to ETD buffer
        self.data_manager.etd_buffer_manager.select_channel_data_and_add(sanitized_board_data_chunk)
        
        # Calculate which buffer should be processed next
        next_buffer_id = self.data_manager._calculate_next_buffer_id_to_process()
        self.logger.info(f"Next buffer ID to process: {next_buffer_id}")

        # Check if we CAN process an epoch (have enough data + right timing)
        can_process, reason, epoch_start_idx_abs, epoch_end_idx_abs = self.data_manager.next_available_epoch_on_round_robin_buffer(next_buffer_id)
        # log epoch start and end indices
        self.logger.info(f"Epoch start index: {epoch_start_idx_abs}")
        self.logger.info(f"Epoch end index: {epoch_end_idx_abs}")
        self.logger.info(f"Can process epoch: {can_process}")
        if not can_process:
            self.logger.info(f"Reason: {reason}")

        if can_process:
            # Only process when we have a complete epoch ready

            # Process the epoch
            self.data_manager.manage_epoch(buffer_id=next_buffer_id, 
                                    epoch_start_idx_abs=epoch_start_idx_abs, 
                                    epoch_end_idx_abs=epoch_end_idx_abs)
            

          
            epoch_start_idx_abs, _ = self.data_manager._get_next_epoch_indices(self.data_manager._calculate_next_buffer_id_to_process())
            # Only trim buffer after successful epoch processing
            # This ensures we don't trim data that hasn't been processed yet
            self.data_manager.etd_buffer_manager.trim_buffer(
                max_next_expected=epoch_start_idx_abs,
            )

        # Log processing statistics
        self.logger.info(f"Collected {self.sample_count} processed samples")


        # Calculate and log basic statistics for monitoring
        