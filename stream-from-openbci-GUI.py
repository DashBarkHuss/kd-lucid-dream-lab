# Summary: This script connects to an OpenBCI Ganglion board via network streaming,
# receives EEG data, and visualizes it as a horizontal graph in the terminal.
# It uses BrainFlow to handle the data stream from the OpenBCI GUI, displaying
# real-time brain activity as a moving dot indicator.

# To connect the OpenBCI GUI to the computer running this script
# 1. System Control Panel -> Ganglion Live
# 2. Select the following settings:
# - Pick Transfer Protocol: BLED112 Dongle
# - BLE Device: select your ganglion board
# - Session Data: BDF+
# - Brainflow Streamer: Network, ip address to 225.1.1.1, port to 6677
# 3. Start Session 
# 4. Start Data Stream
# 5. Remove the filters if you want it to match the logs in this script
# - Filters -> Click "All" to turn the channel icons black which means no filters are applied

# In the terminal you'll see the samples ploted out in a vertical stream.

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np

def get_most_recent_sample(board):
    data = board.get_current_board_data(10)
    eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())
    first_channel = eeg_channels[0]
    return data[first_channel, -1]

def create_horizontal_graph(value, width=50, min_value=-400, max_value=400):
    range_value = max_value - min_value
    position = int((value - min_value) / range_value * width)
    position = max(0, min(position, width - 1))  # Ensure position is within bounds

    graph = " " * position + "•" + " " * (width - position - 1)
    return graph

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.ip_port = 6677
    # params.ip_port_aux = 6678
    params.ip_address = "225.1.1.1"
    # params.ip_address_aux = "225.1.1.1"
    params.master_board = BoardIds.GANGLION_BOARD
    

    board = BoardShim(BoardIds.STREAMING_BOARD, params)

    try:
        print("Board session preparation")
        board.prepare_session()
        print("Board prepared session")
        board.start_stream()
          # Get information about the board
        sampling_rate = BoardShim.get_sampling_rate(BoardIds.GANGLION_BOARD)
        print(f"Sampling Rate: {sampling_rate}")
    
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD)
        print(f"EEG Channels: {eeg_channels}")
    
        accel_channels = BoardShim.get_accel_channels(BoardIds.GANGLION_BOARD)
        print(f"Accelerometer Channels: {accel_channels}")
    
        timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.GANGLION_BOARD)
        print(f"Timestamp Channel: {timestamp_channel}")
    
        time.sleep(1)  # Wait for the board to start streaming
        print("Board started streaming")
        

        # Get data for 10 seconds
        for _ in range(10000):
            data = board.get_current_board_data(10)
    
            # Get the EEG channels
            eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())

            # The first EEG channel
            first_channel = eeg_channels[0]

            # Get the most recent sample from the first channel
            most_recent_sample = data[first_channel, -1]

            # print(f"Most recent sample from channel {first_channel}: {most_recent_sample}")
            sample = get_most_recent_sample(board)
            graph = create_horizontal_graph(sample)
            print(graph)
            # print Value
            # print(most_recent_sample) # if you want to see the value

            time.sleep(0.01)

    except Exception as e:
        print(f"Error occurred: {str(e)}")

    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    main()


