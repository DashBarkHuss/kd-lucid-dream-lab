#!/usr/bin/env python3

"""
Main script for running the multiprocessing board stream with gap handling.

This script provides a complete implementation of the multiprocessing board streaming system,
using BrainFlowChildProcessManager for real-time data acquisition with inter-process communication.
It handles data streaming with gap detection and automatic restart functionality.

Key Features:
- Child process data acquisition via BrainFlowChildProcessManager
- Inter-process communication for robust gap detection
- Automatic stream restart after gaps
- File trimming for gap continuation
- Colored logging for better debugging

Usage:
    python main.py

Configuration:
    - Modify playback_file path to use different data files
    - Configure logging level and format as needed
"""

import sys
import os
# Add the project root to Python path to enable absolute imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

# Now use absolute imports
from gssc_local.realtime_with_restart.board_manager import BoardManager
from gssc_local.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
from gssc_local.realtime_with_restart.core.brainflow_child_process_manager import BrainFlowChildProcessManager
from gssc_local.realtime_with_restart.utils.timestamp_utils import format_elapsed_time
from gssc_local.realtime_with_restart.utils.logging_utils import setup_colored_logger
from gssc_local.realtime_with_restart.utils.file_utils import create_trimmed_csv

import time
import multiprocessing
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds

# Set up logging with colors
logger = setup_colored_logger(__name__)

# Disable BrainFlow's internal logging to avoid interference with our logging
BoardShim.disable_board_logger()

# Timestamp utility functions moved to gssc_local.realtime_with_restart.utils.timestamp_utils


def main(handler_class=ReceivedStreamedDataHandler):
    """Main function that manages the data acquisition and processing.
    
    This function:
    1. Initializes the board with child process streaming via BrainFlowChildProcessManager
    2. Sets up the data handler and visualization components
    3. Manages the main processing loop with inter-process communication
    4. Handles data streaming and gap detection through message passing
    5. Manages cleanup on exit
    
    The function runs until either:
    - No more data is available
    - A gap is detected and handled
    - An error occurs
    
    Args:
        handler_class: The data handler class to instantiate (for dependency injection)
    """
    # Initialize playback file and timestamp tracking
    # original_data_file_path = os.path.join(workspace_root, "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv")
    original_data_file_path = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    # original_data_file_path = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    current_playback_file = original_data_file_path
    
    # Verify input file exists
    if not os.path.isfile(current_playback_file):
        logger.error(f"File not found: {current_playback_file}")
        return

    # Load the CSV file for offset calculation
    original_playback_data = pd.read_csv(current_playback_file, sep='\t', header=None)

    board_id = BoardIds.CYTON_DAISY_BOARD
    board_manager = BoardManager(board_id)
    board_manager.set_board_shim()
    board_timestamp_channel = board_manager.board_timestamp_channel
    board_timestamp_channel_9 = board_manager.board_shim.get_timestamp_channel(board_id)
    received_streamed_data_handler = handler_class(board_manager, logger)

    # Get the PyQt application instance from the visualizer
    qt_app = received_streamed_data_handler.data_manager.visualizer.app

    try:
        # Main processing loop
        while True:
            # Create and start stream manager
            brainflow_child_process_manager = BrainFlowChildProcessManager(current_playback_file, board_id)
            brainflow_child_process_manager.start_stream()
            
            last_good_ts = None
            child_exited_normally = False
            
            # Monitor stream and handle incoming data
            while brainflow_child_process_manager.is_streaming():
                message = brainflow_child_process_manager.get_next_message()
                if message:
                    msg_type, received = message
                    
                    if msg_type == 'start_ts':
                        # Update start timestamp from child
                        brainflow_child_process_manager.start_first_data_ts = float(received) if received is not None else None
                        logger.info(f"Updated start_first_data_ts to: {brainflow_child_process_manager.start_first_data_ts}")
                        
                    elif msg_type == 'last_ts':
                        # Handle gap detection
                        last_good_ts = float(received)
                        logger.info(f"Received last good timestamp: {last_good_ts}")
                        child_exited_normally = True
                        break
                        
                    elif msg_type == 'data':

                        received_streamed_data_handler.process_board_data(received['board_data'])

                    elif msg_type == 'error':
                        # Handle error from child process
                        logger.error(f"Child process reported error: {received}")
                        child_exited_normally = False
                        break
                        
                # Process Qt events to update the GUI
                qt_app.processEvents()
                time.sleep(0.1)
                
            # Clean up stream
            start_first_data_ts = brainflow_child_process_manager.start_first_data_ts  # Store the timestamp before cleanup
            brainflow_child_process_manager.stop_stream()
            brainflow_child_process_manager = None  # Prevent double stop in finally
            
            # Handle abnormal child exit
            if not child_exited_normally:
                logger.error("Child exited without sending a message. Exiting program.")
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
                received_streamed_data_handler.data_manager.csv_manager.save_all_and_cleanup(merge_files=True, merge_output_path="merged_data.csv")
                received_streamed_data_handler.data_manager.validate_saved_csv(original_data_file_path, output_csv_path)
                break

            # Create new trimmed file starting from the gap
            offset = int(next_rows.index[0])
            elapsed_from_start = original_playback_data.iloc[offset, board_timestamp_channel] - start_first_data_ts  # Use stored timestamp
            elapsed_from_last_good_ts = original_playback_data.iloc[offset, board_timestamp_channel] - last_good_ts
            logger.info(f"Restarting from: {offset} | Time from start: {format_elapsed_time(elapsed_from_start)} | Time from last good ts: {format_elapsed_time(elapsed_from_last_good_ts)}")

            trimmed_file = os.path.join(workspace_root, f"data/offset_files/offset_{offset}_{os.path.basename(current_playback_file)}")
            create_trimmed_csv(current_playback_file, trimmed_file, offset)

            # Update playback file for next iteration
            current_playback_file = trimmed_file
            logger.info(f"Updated playback file to: {current_playback_file}")

    except Exception as e:
        import traceback
        logger.error(f"Error in main loop: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Clean up resources
        try:
            if 'received_streamed_data_handler' in locals():
                received_streamed_data_handler.data_manager.cleanup()
            if 'stream_manager' in locals() and brainflow_child_process_manager is not None:
                brainflow_child_process_manager.stop_stream()  # Clean up multiprocessing streams
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Use spawn method for process creation (more reliable than fork)
    multiprocessing.set_start_method('spawn')
    main()