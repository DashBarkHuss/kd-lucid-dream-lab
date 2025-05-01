#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Now use absolute imports
from gssc_local.montage import Montage
from data_buffer_manager import DataBufferManager
from board_manager import BoardManager

# we are using  cyton_daisy_example_gap_stream_reset_and_manage_data.py as a template

# Let's  build out the code to stream open bci data, and match the functionality of @cyton_realtime_round_robin_before_refactor.py plus the ability to start and stop the stream
# Functionality:
# - Start and stop the stream
# - Collect data from the stream
# - Pass data to a buffer manager
# - Pass the data from the buffer manager to the epoch manager/signal manager

# parent.py
import time
import os
import multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
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
    '%(processName)s - %(levelname)s - %(lineno)s - %(message)s'
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

class ReceivedStreamedDataHandler: 
    """Handles processing and storage of incoming EEG data"""
    def __init__(self, input_file: str, board_manager: BoardManager):
        # Initialize buffers to store data chunks and their timestamps
        self.data_buffer = []  # Stores raw EEG data chunks
        self.timestamp_buffer = []  # Stores corresponding timestamps
        self.sample_count = 0  # Total number of samples processed
        self.board_manager = board_manager
        self.data_buffer_manager = DataBufferManager(self.board_manager.board_shim, self.board_manager.sampling_rate, Montage.minimal_sleep_montage())
    
    def process_board_data(self, board_data):
        """Process incoming data chunk and store it"""
        # Store the new data chunk and its timestamps
        self.data_buffer.append(board_data)
        timestamps = board_data[self.board_manager.timestamp_channel]
        self.timestamp_buffer.append(timestamps)

        self.sample_count += board_data.shape[1]  # Increment total sample count

        # Successfully got data
        if self.data_buffer_manager.add_data(board_data): # add data to the full buffer. this only saves the relevant data to the buffer and it validates the data
            # First just save the raw data
            self.data_buffer_manager.save_new_data(board_data)
            
            # Calculate which buffer should be processed next
            next_buffer_id = self.data_buffer_manager._calculate_next_buffer_id_to_process()

            # Check if we CAN process an epoch (have enough data + right timing)
            can_process, reason, epoch_start_idx, epoch_end_idx = self.data_buffer_manager.next_available_epoch_on_buffer(next_buffer_id)

            if can_process:
                # Only process when we have a complete epoch ready
                self.data_buffer_manager.manage_epoch(buffer_id=next_buffer_id, 
                                        epoch_start_idx=epoch_start_idx, 
                                        epoch_end_idx=epoch_end_idx)
        
        # Log processing statistics
        logger.info(f"Processed {self.sample_count} samples")        
        # Calculate and log basic statistics for monitoring
        
def run_board_stream(playback_file, conn):
    """Child process that handles data acquisition from the board"""
    try:
        # Receive initial timestamp from parent process
        msg_type, start_first_data_ts = conn.recv()
        start_first_data_ts = float(start_first_data_ts) if start_first_data_ts is not None else None
        
        # Set up board configuration for playback
        params = BrainFlowInputParams()
        params.board_id = BoardIds.PLAYBACK_FILE_BOARD
        params.file = playback_file
        params.master_board = BoardIds.CYTON_DAISY_BOARD

        # Initialize and start the board
        board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
        board.prepare_session()
        board.config_board('old_timestamps')
        board.start_stream()

        time.sleep(0.1)  # Brief pause to ensure stream is started

        timestamp_channel = BoardShim.get_timestamp_channel(params.master_board)
        last_valid_data_ts = None

        while True:
            # Get new data from the board
            board_data = board.get_board_data()

            # Check for gap in data (empty data array)
            if board_data.shape[1] == 0:
                logger.info(f"Gap detected at {time.time()}, exiting and telling parent to restart. last_valid_data_ts: {last_valid_data_ts}")
                conn.send(('last_ts', last_valid_data_ts))
                logger.info(f"Closed connection at {time.time()}")
                conn.close()
                return

            timestamps = board_data[timestamp_channel]

            # If this is the first data chunk, set the start timestamp
            if start_first_data_ts is None:
                start_first_data_ts = float(timestamps[0])
                # Send the updated start_first_data_ts back to parent
                conn.send(('start_ts', start_first_data_ts))

            last_valid_data_ts = float(timestamps[-1])
            elapsed = last_valid_data_ts - start_first_data_ts
            
            # Send both data and metadata to parent process
            conn.send(('data', {
                'board_data': board_data,
            }))

            time.sleep(1)  # Control the rate of data acquisition

    except Exception as e:
        logger.error(f"Error in board stream: {e}")
        conn.close()

def main():
    """Main function that manages the data acquisition and processing"""
    # Initialize playback file and timestamp tracking
    playback_file = "data/test_data/consecutive_data.csv"
    # playback_file = "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv"
    start_first_data_ts = None  # Keep this at module level for parent process

    # Verify input file exists
    if not os.path.isfile(playback_file):
        logger.error(f"File not found: {playback_file}")
        return

    # Load the CSV file for offset calculation
    original_playback_data = pd.read_csv(playback_file, sep='\t', header=None)

    offset = 0
    board_id = BoardIds.CYTON_DAISY_BOARD

    board_manager = BoardManager(playback_file, board_id)
    board_manager.setup_board()
    timestamp_channel = board_manager.board_shim.get_timestamp_channel(board_id)
    received_streamed_data_handler = ReceivedStreamedDataHandler(playback_file, board_manager)

    # Get the PyQt application instance from the visualizer. uncomment if using the regular matplotlib visualizer
    qt_app = received_streamed_data_handler.data_buffer_manager.visualizer.app

    # Main processing loop
    while True:
        # Create pipe for inter-process communication
        # multiprocessing.Pipe() creates a pair of connected endpoints (two-way communication)
        parent_conn, child_conn = multiprocessing.Pipe()
        
        # Start child process for data acquisition
        p = multiprocessing.Process(target=run_board_stream, args=(playback_file, child_conn))
        p.start()
        
        # Send current timestamp to child process
        parent_conn.send(('start_ts', start_first_data_ts))
        
        last_good_ts = None
        child_exited_normally = False
        
        # Monitor child process and handle incoming data
        while p.is_alive():
            if parent_conn.poll():
                msg_type, received = parent_conn.recv()
                
                if msg_type == 'start_ts':
                    # Update start timestamp from child
                    start_first_data_ts = float(received) if received is not None else None
                    logger.info(f"Updated start_first_data_ts to: {start_first_data_ts}")
                    
                elif msg_type == 'last_ts':
                    # Handle gap detection
                    last_good_ts = float(received)
                    logger.info(f"Received last good timestamp: {last_good_ts}")
                    child_exited_normally = True
                    break
                    
                elif msg_type == 'data':
                    # Process incoming data
                    received_streamed_data_handler.process_board_data(received['board_data'])
                    
            # Process Qt events to update the GUI. uncomment if using the regular matplotlib visualizer
            qt_app.processEvents()
            
            time.sleep(0.1)
            
        # Clean up child process
        if p.is_alive():
            p.terminate()
        p.join()  # Always join to ensure proper cleanup
        
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
            logger.info("No more data after last timestamp. Exiting.")
            break

        # Create new trimmed file starting from the gap
        offset = int(next_rows.index[0])
        elapsed_from_start  = original_playback_data.iloc[offset, timestamp_channel] - start_first_data_ts
        elapsed_from_last_good_ts = original_playback_data.iloc[offset, timestamp_channel] - last_good_ts
        logger.info(f"Restarting from: {offset} | Time from start: {format_elapsed_time(elapsed_from_start)} | Time from last good ts: {format_elapsed_time(elapsed_from_last_good_ts)}")

        trimmed_file = f"data/offset_files/offset_{offset}_{os.path.basename(playback_file)}"
        create_trimmed_csv(playback_file, trimmed_file, offset)

        # Update playback file for next iteration
        playback_file = trimmed_file
        logger.info(f"Updated playback file to: {playback_file}")

if __name__ == "__main__":
    # Use spawn method for process creation (more reliable than fork)
    multiprocessing.set_start_method('spawn')
    main()