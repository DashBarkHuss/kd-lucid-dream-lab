"""
Main script for running the speed-controlled board stream simulation.

This script provides a complete implementation of the speed-controlled board streaming system,
allowing for testing and development with faster playback speeds. It simulates real-time
data streaming from a CSV file at a configurable speed.

Key Features:
- Configurable playback speed through speed_multiplier
- Full pipeline testing including visualization
- Gap detection and handling
- Timestamp tracking and validation
- Colored logging for better debugging
- **NEW: Sleep stage event system** - responds to sleep stage detections in real-time

Usage:
    python main_speed_controlled_stream.py

Event System:
    - Events are ENABLED by default - you'll see sleep stage event messages in the console
    - To disable events, change enable_events=False in the main() call at the bottom
    - Example events: "REM SLEEP DETECTED!", stage counters, confidence alerts
    - Customize handlers in create_event_dispatcher() function

Configuration:
    - Modify playback_file path to use different data files
    - Adjust speed_multiplier in SpeedControlledBoardManager initialization to change playback speed
    - Configure logging level and format as needed
    - Add/remove event handlers in create_event_dispatcher()
"""

import sys
import os
# Add the project root to Python path to enable absolute imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

# Now use absolute imports
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.realtime_with_restart.speed_controlled_board_manager import SpeedControlledBoardManager
from sleep_scoring_toolkit.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
from sleep_scoring_toolkit.realtime_with_restart.utils.logging_utils import setup_colored_logger
from sleep_scoring_toolkit.realtime_with_restart.utils.file_utils import create_trimmed_csv
from sleep_scoring_toolkit.realtime_with_restart.utils.session_utils import generate_session_timestamp, save_session_csv_files, setup_signal_handlers
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import RawBoardDataWithKeys

# Event system imports (for sleep stage event handling)
from sleep_scoring_toolkit.realtime_with_restart.event_system import EventDispatcher
from sleep_scoring_toolkit.experiments.handlers import handler
# Note: timestamp utilities available if needed for future logging

import time
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

# Set up logging with colors
logger = setup_colored_logger(__name__)

# Global variables to track data handler and session info for signal handlers
received_streamed_data_handler = None
session_timestamp = None

# Disable BrainFlow's internal logging to avoid interference with our logging
BoardShim.disable_board_logger()

# Timestamp utility functions moved to gssc_local.realtime_with_restart.utils.timestamp_utils




def create_event_dispatcher():
    """Create and configure the event dispatcher with unified handler.
    
    Returns:
        EventDispatcher: Configured event dispatcher with unified handler
    """
    dispatcher = EventDispatcher()
    
    # Register single unified handler
    dispatcher.register_handler(handler)
    
    logger.info("Event dispatcher configured with unified handler")
    
    # Print registered handlers
    stats = dispatcher.get_stats()
    logger.info(f"Registered handlers: {stats['registered_handlers']}")
    
    return dispatcher


def main(handler_class=ReceivedStreamedDataHandler, enable_events=True):
    """Main function that manages the data acquisition and processing.
    
    This function:
    1. Generates a session timestamp for unique file naming
    2. Initializes the mock board with the specified data file
    3. Sets up the data handler and visualization
    4. Manages the main processing loop
    5. Handles data streaming and gap detection
    6. Manages cleanup on exit
    
    The function runs until either:
    - No more data is available
    - A gap is detected
    - An error occurs
    """
    # Generate session timestamp for unique file naming
    global session_timestamp
    session_timestamp = generate_session_timestamp()
    logger.info(f"🕐 Session started at: {session_timestamp}")
    """Main function that manages the data acquisition and processing.
    
    This function:
    1. Initializes the mock board with the specified data file
    2. Sets up the data handler and visualization
    3. Manages the main processing loop
    4. Handles data streaming and gap detection
    5. Manages cleanup on exit
    
    The function runs until either:
    - No more data is available
    - A gap is detected
    - An error occurs
    """
    # Initialize playback file and timestamp tracking
    # original_data_file = os.path.join(workspace_root, "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv")
    # original_data_file = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    original_data_file = os.path.join(workspace_root, "data/test_data/gapped_data.csv")  # Test gap detection
    playback_file = original_data_file
    
    # Verify input file exists
    if not os.path.isfile(playback_file):
        logger.error(f"File not found: {playback_file}")
        return

    # Load the CSV file for offset calculation
    original_playback_data = pd.read_csv(playback_file, sep='\t', header=None)

    # Initialize board and handler
    board_id = BoardIds.CYTON_DAISY_BOARD  # Keep for consistency with main.py
    # Use SpeedControlledBoardManager with moderate speed for gap testing
    mock_board_manager_speed_control = SpeedControlledBoardManager(playback_file, speed_multiplier=10.0)
    mock_board_manager_speed_control.set_board_shim()     
    board_timestamp_channel = mock_board_manager_speed_control.board_timestamp_channel
    
    # Create montage for proper initialization
    montage = Montage.default_sleep_montage()
    # montage = Montage.eog_only_montage()
    
    # Create event dispatcher if events are enabled
    event_dispatcher = None
    if enable_events:
        event_dispatcher = create_event_dispatcher()
        logger.info("🎯 Event system enabled - sleep stage events will be processed!")
    else:
        logger.info("Event system disabled - running in standard mode")
    
    global received_streamed_data_handler
    received_streamed_data_handler = handler_class(mock_board_manager_speed_control, logger, montage, event_dispatcher, session_timestamp)
    
    # Setup signal handlers after we have the data handler
    setup_signal_handlers(
        received_streamed_data_handler,
        session_timestamp,
    )

    # Get the PyQt application instance from the visualizer
    qt_app = received_streamed_data_handler.data_manager.visualizer.app

    try:
        # Main processing loop
        while True:
            # Create and start stream
            mock_board_manager_speed_control.start_stream()
            logger.info("Started speed-controlled board stream")
            
            last_good_ts = None
            start_first_data_ts = None
            data_processing_completed = False
            
            # Monitor stream and handle incoming data
            while True:
                # Get new data chunk (raw numpy array from board manager)
                raw_board_data = mock_board_manager_speed_control.get_new_data()
                
                # Wrap raw data for explicit keying
                board_data_keyed = RawBoardDataWithKeys(raw_board_data)
                
                if board_data_keyed.size > 0:
                    # If this is the first data chunk, set the start timestamp
                    if start_first_data_ts is None and board_timestamp_channel is not None:
                        start_first_data_ts = float(board_data_keyed.get_by_key(board_timestamp_channel)[0])
                        logger.info(f"Set start_first_data_ts to: {start_first_data_ts}")
                    
                    # Process the data through the pipeline
                    received_streamed_data_handler.process_board_data_chunk(board_data_keyed)
                    
                    # Update last good timestamp for gap detection
                    if board_timestamp_channel is not None:
                        last_good_ts = float(board_data_keyed.get_by_key(board_timestamp_channel)[-1])

                else:
                    # No more data - equivalent to gap detection in main.py
                    logger.info("No more data to process")
                    data_processing_completed = True
                    break
                    
                # Process Qt events to update the GUI
                qt_app.processEvents()
                time.sleep(0.1)
                
            # Clean up stream (no MultiprocessStreamManager to clean up for direct board access)
            # start_first_data_ts already stored as local variable
            
            # Handle processing status (similar to child_exited_normally in main.py)
            if not data_processing_completed:
                logger.error("Data processing failed. Exiting program.")
                break

            if last_good_ts is None:
                logger.error("No valid timestamp received. Exiting.")
                break

            # Calculate new offset after gap
            timestamps = original_playback_data.iloc[:, board_timestamp_channel]
            next_rows = timestamps[timestamps > last_good_ts]
            
            if next_rows.empty:
                logger.info("No more data after last timestamp. Saving csv and exiting.")
                # validate the saved csv
                logger.info(f"Main csv buffer path before final save: {received_streamed_data_handler.data_manager.csv_manager.main_csv_path}")        
                output_csv_path = received_streamed_data_handler.data_manager.csv_manager.main_csv_path
                save_session_csv_files(received_streamed_data_handler, session_timestamp)
                received_streamed_data_handler.data_manager.validate_saved_csv(original_data_file, output_csv_path)
                break
                
            # Create new trimmed file starting from the gap
            # This simulates continuing after a gap
            offset = int(next_rows.index[0])
            trimmed_file = os.path.join(workspace_root, f"data/offset_files/offset_{offset}_{os.path.basename(playback_file)}")
            create_trimmed_csv(playback_file, trimmed_file, offset)

            # Update playback file for next iteration
            playback_file = trimmed_file
            logger.info(f"Updated playback file to: {playback_file}")
            
            # Update the board manager's file path (no need to create new instance like main.py)
            # This ensures we continue from the correct point
            mock_board_manager_speed_control.file_path = playback_file
            # Reload the data from the new file
            mock_board_manager_speed_control.file_data = pd.read_csv(playback_file, sep='\t', header=None, dtype=float)
            mock_board_manager_speed_control.current_position = 0
            
            # Reset gap detection state for clean restart
            mock_board_manager_speed_control.in_gap_mode = False
            mock_board_manager_speed_control.gap_start_real_time = None
            mock_board_manager_speed_control.gap_duration_seconds = None
            mock_board_manager_speed_control.expected_timestamp = None
            
            mock_board_manager_speed_control.start_stream()  # Reset stream position
            
            # No need to recreate handlers - they can reuse the same board manager

    except Exception as e:
        import traceback
        logger.error(f"Error in main loop: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Clean up resources and save CSV as fallback
        try:
            # Try to save CSV data as fallback (signal handlers handle normal termination)
            try:
                save_session_csv_files(received_streamed_data_handler, session_timestamp)
                logger.info("✅ CSV files saved and merged successfully")
            except Exception as save_error:
                logger.error(f"Failed to save CSV in finally block: {save_error}")
                
            received_streamed_data_handler.data_manager.cleanup()
            # No multiprocess_stream_manager cleanup needed (uses SpeedControlledBoardManager directly)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # No multiprocessing needed for mock board
    main() 