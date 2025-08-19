#!/usr/bin/env python3

"""
Main script for running real-time OpenBCI streaming with GSSC sleep classification.

This script connects directly to live OpenBCI data streams (via OpenBCI GUI) and 
processes the data through the full pipeline including visualization, sleep stage 
classification, and CSV export.

Key Features:
- Real-time data streaming from OpenBCI hardware via network
- Full processing pipeline including visualization and sleep classification
- Automatic CSV data export with sleep stage annotations
- Colored logging for better debugging

Setup:
1. Open OpenBCI GUI
2. System Control Panel -> Cyton + Daisy Live  
3. Set Brainflow Streamer to Network: IP 225.1.1.1, Port 6677
4. Start Session -> Start Data Stream
5. Run this script

Usage:
    python main_realtime_stream.py

Configuration:
    - Modify network parameters (IP/port) if using different OpenBCI GUI settings
    - Configure logging level and format as needed
"""

import sys
import os
# Add the project root to Python path to enable absolute imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

# Now use absolute imports
from sleep_scoring_toolkit.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
from sleep_scoring_toolkit.realtime_with_restart.utils.logging_utils import setup_colored_logger
from sleep_scoring_toolkit.realtime_with_restart.utils.session_utils import generate_session_timestamp, save_session_csv_files, setup_signal_handlers
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import RawBoardDataWithKeys
from sleep_scoring_toolkit.montage import Montage

import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Set up logging with colors
logger = setup_colored_logger(__name__)

# Global variables to track resources for signal handlers
received_streamed_data_handler = None
streaming_board_manager = None
session_timestamp = None

# Disable BrainFlow's internal logging to avoid interference with our logging
BoardShim.disable_board_logger()




class StreamingBoardManager:
    """Standalone board manager that connects to live OpenBCI streaming data."""
    
    def __init__(self, board_id=BoardIds.CYTON_DAISY_BOARD, ip_address="225.1.1.1", ip_port=6677):
        """Initialize streaming board manager.
        
        Args:
            board_id: The OpenBCI board type
            ip_address: Network IP for streaming (must match OpenBCI GUI setting)
            ip_port: Network port for streaming (must match OpenBCI GUI setting)
        """
        self.board_id = board_id
        self.ip_address = ip_address
        self.ip_port = ip_port
        
        # Initialize board attributes
        self.board_shim = None
        self.sampling_rate = None
        self.board_timestamp_channel = None
        self.eeg_channels = None

        
    def set_board_shim(self):
        """Set up the board shim with streaming parameters."""
        params = BrainFlowInputParams()
        params.ip_address = self.ip_address
        params.ip_port = self.ip_port
        params.master_board = self.board_id
        
        self.board_shim = BoardShim(BoardIds.STREAMING_BOARD, params)
        self._set_board_info()
        
    def _set_board_info(self):
        """Set board information using the board ID for correct channel mapping."""
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.board_timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
    def start_stream(self):
        """Start the streaming session."""
        if not self.board_shim.is_prepared():
            self.board_shim.prepare_session()
        self.board_shim.start_stream()
        
    def get_new_raw_data_chunk(self):
        """Get new data from the stream.
        
        Returns:
            numpy.ndarray: Raw board data from BrainFlow
        """
        new_raw_board_chunk = self.board_shim.get_board_data()  # Get all new data since last call
        
        # Debug logging to investigate duplicate timestamps
        if new_raw_board_chunk.size > 0:
            logger.info(f"BrainFlow returned {new_raw_board_chunk.shape[1]} samples, first timestamp: {new_raw_board_chunk[self.board_timestamp_channel][0]:.6f}, last timestamp: {new_raw_board_chunk[self.board_timestamp_channel][-1]:.6f}")
        else:
            logger.info("BrainFlow returned no data")
            
        return new_raw_board_chunk
        
    def stop_stream(self):
        """Stop the streaming session."""
        if self.board_shim.is_prepared():
            self.board_shim.stop_stream()
            self.board_shim.release_session()


def main(handler_class=ReceivedStreamedDataHandler, montage=None):
    """Main function that manages the real-time data acquisition and processing.
    
    This function:
    1. Generates a session timestamp for unique file naming
    2. Initializes the streaming board connection to OpenBCI GUI
    3. Sets up the data handler and visualization
    4. Manages the main processing loop
    5. Handles data streaming and basic error conditions
    6. Manages cleanup and CSV export on exit
    
    The function runs until either:
    - A keyboard interrupt (Ctrl+C)
    - An error occurs
    - The stream stops providing data for an extended period
    
    Args:
        handler_class: The data handler class to instantiate (for dependency injection)
        montage: Montage configuration to use

    """
    # Generate session timestamp for unique file naming
    global session_timestamp
    session_timestamp = generate_session_timestamp()
    logger.info(f"🕐 Session started at: {session_timestamp}")
    logger.info("Starting real-time OpenBCI stream processor")
    
    # Initialize streaming board manager
    global streaming_board_manager, received_streamed_data_handler
    streaming_board_manager = StreamingBoardManager()
    streaming_board_manager.set_board_shim()
    board_timestamp_channel = streaming_board_manager.board_timestamp_channel
    
    # Create data handler with optional montage configuration
    received_streamed_data_handler = handler_class(streaming_board_manager, logger, montage, session_timestamp=session_timestamp)
    
    # Setup signal handlers after we have the resources
    setup_signal_handlers(
        received_streamed_data_handler,
        session_timestamp,
        streaming_board_manager
    )

    # Get the PyQt application instance from the visualizer
    qt_app = received_streamed_data_handler.data_manager.visualizer.app

    try:
        # Connect and start streaming
        logger.info("Connecting to OpenBCI stream...")
        streaming_board_manager.start_stream()
        logger.info("Connected! Waiting for data...")
        
        # Wait for buffer to fill
        time.sleep(2)
        
        start_first_data_ts = None
        no_data_count = 0
        max_no_data_cycles = 20  # Exit after 10 seconds of no data (20 * 0.5s)
        
        # Main processing loop
        while True:
            # Get new data chunk (raw numpy array from board manager)
            raw_board_data = streaming_board_manager.get_new_raw_data_chunk()
            
            # Wrap raw data for explicit keying
            board_data_keyed = RawBoardDataWithKeys(raw_board_data)
            
            if board_data_keyed.size > 0:


                #  all inter batch data sanitization like removing old timestamps, needs to happen where the last saved timestamp is in scope               
                # Use sanitized and reordered data for processing
                # Reset no-data counter
                no_data_count = 0
                
                # If this is the first data chunk, set the start timestamp
                if start_first_data_ts is None and board_timestamp_channel is not None:
                    start_first_data_ts = float(board_data_keyed.get_by_key(board_timestamp_channel)[0])
                    logger.info(f"Started processing data at timestamp: {start_first_data_ts}")
                
                # Process the data through the pipeline
                received_streamed_data_handler.process_board_data_chunk(board_data_keyed)
                
            else:
                # No data received
                no_data_count += 1
                if no_data_count % 10 == 0:  # Log every 5 seconds
                    logger.warning(f"No data received for {no_data_count * 0.5:.1f} seconds...")
                
                if no_data_count >= max_no_data_cycles:
                    logger.error("No data received for extended period. Check OpenBCI GUI stream.")
                    # Save CSV data before exiting due to no data
                    logger.info("Saving CSV data before exit...")
                    save_session_csv_files(received_streamed_data_handler, session_timestamp)
                    break
                    
            # Process Qt events to update the GUI
            qt_app.processEvents()
            time.sleep(0.5)  # Update every 0.5 seconds
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Stopping stream...")
    except Exception as e:
        import traceback
        logger.error(f"Error in main loop: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Clean up resources
        try:
            logger.info("Cleaning up and saving data...")
            
            # Try to save CSV data as fallback (signal handlers handle normal termination)
            try:
                save_session_csv_files(received_streamed_data_handler, session_timestamp)
                logger.info("✅ CSV files saved and merged successfully")
            except Exception as save_error:
                logger.error(f"Failed to save CSV in finally block: {save_error}")
            
            # Clean up resources
            received_streamed_data_handler.data_manager.cleanup()
            
            if streaming_board_manager is not None:
                streaming_board_manager.stop_stream()
            logger.info("Cleanup complete.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    main(montage=Montage.eog_only_montage())