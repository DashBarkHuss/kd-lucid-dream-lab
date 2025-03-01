"""
This script demonstrates how to use BrainFlow to playback EEG data from a CSV file, insert markers at 
specific timestamps, and extract those markers with their corresponding timestamps. It initializes a 
playback session using a Ganglion board configuration, inserts 10 sequential markers while streaming, 
and then retrieves the data with markers. The script converts marker timestamps to human-readable 
datetime strings and prints them. Finally, it demonstrates how to convert the data to a pandas DataFrame 
for analysis and shows proper data serialization using BrainFlow's built-in file operations.
"""

import argparse
import time
from datetime import datetime
from brainflow.data_filter import DataFilter
import pandas as pd
import numpy as np


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets


def main():
    BoardShim.enable_dev_board_logger()


    params = BrainFlowInputParams()
    # play from file
    # params board_id PLAYBACK_FILE_BOARD
    params.board_id = BoardIds.PLAYBACK_FILE_BOARD
    params.master_board = BoardIds.GANGLION_BOARD
    # I needed to use the full path or the path from the root because I run the script from the root directory using .vscode/launch.json   
    params.file = "data/BrainFlow-RAW.csv"
    # params.file = "/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/playback/BrainFlow-RAW.csv"


    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

 
    board.prepare_session()

    board.start_stream()
    
    for i in range(10):
        time.sleep(1)
        board.insert_marker(i + 1) # pass in a number value here as the marker
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    marker_channel = BoardShim.get_marker_channel(BoardIds.GANGLION_BOARD)
    data_marker = data[marker_channel]
    # find the indexes of the marker channels that aren't zero
    marker_indexes = [i for i, x in enumerate(data_marker) if x != 0]
    # print out thetime stamps of the markers
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.GANGLION_BOARD)
    timestamps = data[timestamp_channel][marker_indexes]
    # Convert to time strings
    time_strings = []
    for ts in timestamps:
        # Convert to datetime object
        dt = datetime.fromtimestamp(ts)
        # Format as string
        time_string = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        time_strings.append(time_string)

    print(time_strings)
        # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    df = pd.DataFrame(np.transpose(data))
    print('Data From the Board')
    print(df.head(10))

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    restored_data = DataFilter.read_file('test.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))
    print('Data From the File')
    print(restored_df.head(10))


if __name__ == "__main__":
    main()