#!/usr/bin/env python3

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

Usage:
    python main_speed_controlled_stream.py

Configuration:
    - Modify playback_file path to use different data files
    - Adjust speed_multiplier in SpeedControlledBoardManager initialization to change playback speed
    - Configure logging level and format as needed
"""

import sys
import os
# Add the project root to Python path to enable absolute imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

# Now use absolute imports
from gssc_local.montage import Montage
from gssc_local.realtime_with_restart.speed_controlled_board_manager import SpeedControlledBoardManager
from gssc_local.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
from gssc_local.realtime_with_restart.utils.logging_utils import setup_colored_logger
from gssc_local.realtime_with_restart.utils.file_utils import create_trimmed_csv
# Note: timestamp utilities available if needed for future logging

import time
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

# Set up logging with colors
logger = setup_colored_logger(__name__)

# Disable BrainFlow's internal logging to avoid interference with our logging
BoardShim.disable_board_logger()

# Timestamp utility functions moved to gssc_local.realtime_with_restart.utils.timestamp_utils


def main(handler_class=ReceivedStreamedDataHandler):
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
    original_data_file = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    # original_data_file = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    playback_file = original_data_file
    
    # Verify input file exists
    if not os.path.isfile(playback_file):
        logger.error(f"File not found: {playback_file}")
        return

    # Load the CSV file for offset calculation
    # This is used to handle gaps in the data
    original_playback_data = pd.read_csv(playback_file, sep='\t', header=None)

    # Initialize board and handler
    board_id = BoardIds.CYTON_DAISY_BOARD  # Keep for consistency with main.py
    # Use SpeedControlledBoardManager with 100x speed for fast testing
    board_manager = SpeedControlledBoardManager(playback_file, speed_multiplier=2000.0)
    board_manager.setup_board()
    board_timestamp_channel = board_manager.board_timestamp_channel
    received_streamed_data_handler = handler_class(board_manager, logger)

    # Configure buffer sizes and file paths for periodic saving
    # Get the PyQt application instance from the visualizer
    # This is needed to process GUI events during streaming
    qt_app = received_streamed_data_handler.data_manager.visualizer.app

    try:
        # Main processing loop
        while True:
            # Create and start stream
            board_manager.start_stream()
            logger.info("Started speed-controlled board stream")
            
            last_good_ts = None
            start_first_data_ts = None
            data_processed_successfully = False
            
            # Monitor stream and handle incoming data
            while True:
                # Get new data chunk
                new_data = board_manager.get_new_data()
                
                if new_data.size > 0:
                    # If this is the first data chunk, set the start timestamp
                    if start_first_data_ts is None and board_timestamp_channel is not None:
                        start_first_data_ts = float(new_data[board_timestamp_channel][0])
                        logger.info(f"Set start_first_data_ts to: {start_first_data_ts}")
                    
                    # Process the data through the pipeline
                    received_streamed_data_handler.process_board_data(new_data)
                    
                    # Update last good timestamp for gap detection
                    if board_timestamp_channel is not None:
                        last_good_ts = float(new_data[board_timestamp_channel][-1])

                else:
                    # No more data - equivalent to gap detection in main.py
                    logger.info("No more data to process")
                    data_processed_successfully = True
                    break
                    
                # Process Qt events to update the GUI
                # This ensures the visualization stays responsive
                qt_app.processEvents()
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
            # Clean up stream (no StreamManager to clean up for direct board access)
            # start_first_data_ts already stored as local variable
            
            # Handle processing status (similar to child_exited_normally in main.py)
            if not data_processed_successfully:
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
                # Validate the saved csv
                logger.info(f"Main csv buffer path before final save: {received_streamed_data_handler.data_manager.csv_manager.main_csv_path}")        
                output_csv_path = received_streamed_data_handler.data_manager.csv_manager.main_csv_path
                received_streamed_data_handler.data_manager.csv_manager.save_all_and_cleanup(merge_files=True, merge_output_path="merged_data.csv")
                received_streamed_data_handler.data_manager.validate_saved_csv(original_data_file, output_csv_path)
                break
                
            # Create a new file for the remaining data
            # This simulates continuing after a gap
            new_start_index = next_rows.index[0]
            new_file_path = os.path.join(
                os.path.dirname(playback_file),
                f"gap_continuation_{new_start_index}.csv"
            )
            
            # Save the remaining data to a new file
            original_playback_data.iloc[new_start_index:].to_csv(
                new_file_path, sep='\t', header=False, index=False
            )
            
            # Update the playback file to continue from the gap
            playback_file = new_file_path
            logger.info(f"Created continuation file: {new_file_path}")
            
            # Reinitialize the board manager with the new file
            # This ensures we continue from the correct point
            board_manager = SpeedControlledBoardManager(playback_file, speed_multiplier=100.0)
            board_manager.setup_board()
            received_streamed_data_handler = ReceivedStreamedDataHandler(board_manager, logger)
            qt_app = received_streamed_data_handler.data_manager.visualizer.app

    except Exception as e:
        import traceback
        logger.error(f"Error in main loop: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Clean up resources
        try:
            if 'received_streamed_data_handler' in locals():
                received_streamed_data_handler.data_manager.cleanup()
            if 'board_manager' in locals():
                board_manager.release()
            # No stream_manager cleanup needed (uses SpeedControlledBoardManager directly)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # No multiprocessing needed for mock board
    main() 