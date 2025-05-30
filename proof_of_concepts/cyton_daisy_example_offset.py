"""
This script trims a BrainFlow CSV data file to start from a specified sample index
and plays back the trimmed file using BrainFlow's PlaybackFileBoard.

Workflow:
1. It checks that the original CSV file exists.
2. It creates a trimmed version of the file, skipping the first `offset` samples.
3. It configures BrainFlow to use the trimmed file for playback.
4. It starts and stops the BrainFlow data stream, verifying that the playback started from the expected sample.
5. It automatically deletes the trimmed temporary file after the session ends.

Usage:
- Useful for testing BrainFlow playback from a non-zero starting point.
- Intended for development, debugging, or test automation scenarios.
"""

import time
import os
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

def create_trimmed_csv(input_file, output_file, skip_samples):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            if idx >= skip_samples:
                outfile.write(line)

def main():
    # Setup
    BoardShim.enable_dev_board_logger()
    BoardShim.set_log_level(LogLevels.LEVEL_TRACE)

    original_file = os.path.join("data", "test_data", "consecutive_data.csv")
    offset = 5
    offset_dir = os.path.join("data", "offset_files")
    os.makedirs(offset_dir, exist_ok=True)

    # Check original file exists
    if not os.path.isfile(original_file):
        print(f"ERROR: {original_file} not found.")
        return

    # Build offset filename
    base_name = os.path.splitext(os.path.basename(original_file))[0]  # -> "consecutive_data"
    offset_file_name = f"start_at_{offset}_{base_name}.csv"
    offset_file_path = os.path.join(offset_dir, offset_file_name)

    # Create trimmed CSV
    create_trimmed_csv(original_file, offset_file_path, offset)

    # Setup BrainFlow params
    params = BrainFlowInputParams()
    params.serial_port = ""
    params.file = offset_file_path
    params.board_id = BoardIds.PLAYBACK_FILE_BOARD
    params.master_board = BoardIds.CYTON_DAISY_BOARD
    params.playback_file_max_count = 1
    params.playback_speed = 1

    board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

    try:
        print("Preparing session...")
        board_shim.prepare_session()

        print("Starting stream...")
        board_shim.start_stream()

        print("Streaming for 2 seconds...")
        time.sleep(2)

        print("Stopping stream...")
        board_shim.stop_stream()

        data = board_shim.get_board_data()
        print(f"Retrieved {data.shape[1]} data points")

        expected_values = np.array([offset + 1, offset + 2, offset + 3, offset + 4, offset + 5])
        actual_values = data[1][:5]

        assert np.allclose(actual_values, expected_values), f"Offset test failed: Unexpected first 5 values: {actual_values}"

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Releasing session...")
        board_shim.release_session()

        try:
            if os.path.isfile(offset_file_path):
                os.remove(offset_file_path)
                print(f"Deleted temporary file {offset_file_path}")
        except Exception as e:
            print(f"Could not delete temporary file: {e}")

if __name__ == "__main__":
    main()