#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

# Now use absolute imports
from gssc_local.montage import Montage
from gssc_local.realtime_with_restart.data_manager import DataManager
from gssc_local.realtime_with_restart.board_manager import BoardManager
from gssc_local.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
from gssc_local.realtime_with_restart.core.stream_manager import StreamManager

import time
import multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from brainflow.board_shim import BoardShim, BoardIds
import logging

# ANSI color codes for logging
class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Custom formatter that adds colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Color the process name
        if 'MainProcess' in record.processName:
            record.processName = f"{LogColors.BLUE}{record.processName}{LogColors.ENDC}"
        else:
            record.processName = f"{LogColors.CYAN}{record.processName}{LogColors.ENDC}"
        
        # Color the log level
        if record.levelno == logging.INFO:
            record.levelname = f"{LogColors.GREEN}{record.levelname}{LogColors.ENDC}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{LogColors.YELLOW}{record.levelname}{LogColors.ENDC}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{LogColors.RED}{record.levelname}{LogColors.ENDC}"
        
        # Add line number in yellow
        record.lineno = f"{LogColors.YELLOW}L{record.lineno}{LogColors.ENDC}"
        
        return super().format(record)

# Set up logging with colors
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    '%(processName)s - %(levelname)s - L%(lineno)s - %(message)s'
))
logger.addHandler(console_handler)

# Disable BrainFlow's internal logging to avoid interference with our logging
BoardShim.disable_board_logger()

def format_timestamp(ts):
    """Convert Unix timestamp to human-readable format in HST (Hawaii Standard Time)"""
    if ts is None:
        return "None"
    # Convert Unix timestamp to datetime in UTC
    utc_time = datetime.fromtimestamp(ts, timezone.utc)
    # Convert to Hawaii time (UTC-10)
    hawaii_time = utc_time - timedelta(hours=10)
    return hawaii_time.strftime('%Y-%m-%d %I:%M:%S %p HST')

def format_elapsed_time(seconds):
    """Format elapsed time in HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def create_trimmed_csv(input_file, output_file, skip_samples):
    """Create a new CSV file starting from the specified sample offset"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            if idx >= skip_samples:
                outfile.write(line)

def main():
    """Main function that manages the data acquisition and processing"""
    # Initialize playback file and timestamp tracking
    # original_data_file = os.path.join(workspace_root, "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv")
    # original_data_file = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    original_data_file = os.path.join(workspace_root, "data/test_data/consecutive_data.csv")
    playback_file = original_data_file
    
    # Verify input file exists
    if not os.path.isfile(playback_file):
        logger.error(f"File not found: {playback_file}")
        return

    # Load the CSV file for offset calculation
    original_playback_data = pd.read_csv(playback_file, sep='\t', header=None)

    board_id = BoardIds.CYTON_DAISY_BOARD
    board_manager = BoardManager(playback_file, board_id)
    board_manager.setup_board()
    timestamp_channel = board_manager.board_shim.get_timestamp_channel(board_id)
    received_streamed_data_handler = ReceivedStreamedDataHandler(board_manager, logger)

    # Get the PyQt application instance from the visualizer
    qt_app = received_streamed_data_handler.data_manager.visualizer.app

    try:
        # Main processing loop
        while True:
            # Create and start stream manager
            stream_manager = StreamManager(playback_file, board_id)
            stream_manager.start_stream()
            
            last_good_ts = None
            child_exited_normally = False
            
            # Monitor stream and handle incoming data
            while stream_manager.is_streaming():
                message = stream_manager.get_next_message()
                if message:
                    msg_type, received = message
                    
                    if msg_type == 'start_ts':
                        # Update start timestamp from child
                        stream_manager.start_first_data_ts = float(received) if received is not None else None
                        logger.info(f"Updated start_first_data_ts to: {stream_manager.start_first_data_ts}")
                        
                    elif msg_type == 'last_ts':
                        # Handle gap detection
                        last_good_ts = float(received)
                        logger.info(f"Received last good timestamp: {last_good_ts}")
                        child_exited_normally = True
                        break
                        
                    elif msg_type == 'data':
                        # Process incoming data
                        received_streamed_data_handler.process_board_data(received['board_data'])
                        
                # Process Qt events to update the GUI
                qt_app.processEvents()
                time.sleep(0.1)
                
            # Clean up stream
            start_first_data_ts = stream_manager.start_first_data_ts  # Store the timestamp before cleanup
            stream_manager.stop_stream()
            stream_manager = None  # Prevent double stop in finally
            
            # Handle abnormal child exit
            if not child_exited_normally:
                logger.error("Child exited without sending a message. Exiting program.")
                break

            if last_good_ts is None:
                logger.error("No valid timestamp received. Exiting.")
                break

            # Calculate new offset after gap
            timestamps = original_playback_data.iloc[:, timestamp_channel]
            next_rows = timestamps[timestamps > last_good_ts]

            if next_rows.empty:
                logger.info("No more data after last timestamp. Saving csv and exiting.")
                output_csv_path = os.path.join(workspace_root, "data/test_data/reconstructed_data.csv")
                received_streamed_data_handler.data_manager.output_csv_path = output_csv_path
                received_streamed_data_handler.data_manager.save_to_csv(output_csv_path)
                # validate the saved csv
                received_streamed_data_handler.data_manager.validate_saved_csv(original_data_file)
                break

            # Create new trimmed file starting from the gap
            offset = int(next_rows.index[0])
            elapsed_from_start = original_playback_data.iloc[offset, timestamp_channel] - start_first_data_ts  # Use stored timestamp
            elapsed_from_last_good_ts = original_playback_data.iloc[offset, timestamp_channel] - last_good_ts
            logger.info(f"Restarting from: {offset} | Time from start: {format_elapsed_time(elapsed_from_start)} | Time from last good ts: {format_elapsed_time(elapsed_from_last_good_ts)}")

            trimmed_file = os.path.join(workspace_root, f"data/offset_files/offset_{offset}_{os.path.basename(playback_file)}")
            create_trimmed_csv(playback_file, trimmed_file, offset)

            # Update playback file for next iteration
            playback_file = trimmed_file
            logger.info(f"Updated playback file to: {playback_file}")

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
            if 'stream_manager' in locals() and stream_manager is not None:
                stream_manager.stop_stream()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Use spawn method for process creation (more reliable than fork)
    multiprocessing.set_start_method('spawn')
    main()