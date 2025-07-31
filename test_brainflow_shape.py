from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np

def main():
    # Enable BrainFlow logger
    BoardShim.enable_dev_board_logger()
    
    # Initialize board parameters
    params = BrainFlowInputParams()
    params.serial_port = "" # Empty for playback board
    params.file = "data/test_data/segmend_of_real_data.csv"  # Path to your recording file
    params.board_id = BoardIds.PLAYBACK_FILE_BOARD
    params.master_board = BoardIds.CYTON_DAISY_BOARD
    
    # Create board instance
    board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
    
    try:
        # Prepare session
        print("Preparing session...")
        board_shim.prepare_session()
        
        # Start streaming
        print("Starting stream...")
        board_shim.start_stream()
        
        # Get some data
        data = board_shim.get_current_board_data(10)
        
        # Print debug information
        print("\n[DEBUG] BrainFlow Data Shape Test:")
        print(f"Data shape: {data.shape}")
        print(f"Number of rows (shape[0]): {data.shape[0]}")
        print(f"Number of columns (shape[1]): {data.shape[1]}")
        print(f"First row (first 5 values): {data[0][:5]}")
        print(f"Second row (first 5 values): {data[1][:5]}")
        
        # Get channel information
        eeg_channels = BoardShim.get_eeg_channels(board_shim.get_board_id())
        print(f"\nEEG channels: {eeg_channels}")
        print(f"Number of EEG channels: {len(eeg_channels)}")
        
        # Get total number of channels
        total_channels = BoardShim.get_num_rows(board_shim.get_board_id())
        print(f"\nTotal channels (from get_num_rows): {total_channels}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always release the session
        print("\nReleasing session...")
        board_shim.release_session()
        print("Session released")

if __name__ == "__main__":
    main() 