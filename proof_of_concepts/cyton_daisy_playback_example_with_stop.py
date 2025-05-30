import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter

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
        
        # Let it run for 5 seconds
        print("Streaming for 5 seconds...")
        time.sleep(5)
        
        # Stop streaming
        print("Stopping stream...")
        board_shim.stop_stream()
        
        # Get the data
        data = board_shim.get_board_data()
        print(f"Retrieved {data.shape[1]} data points")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always release the session
        print("Releasing session...")
        board_shim.release_session()
        print("Session released")

if __name__ == "__main__":
    main() 