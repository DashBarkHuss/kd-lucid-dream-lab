"""
Simple real-time streaming example for OpenBCI data.
Receives stream and provides basic visual feedback in terminal.

Setup:
1. Open OpenBCI GUI
2. System Control Panel -> Cyton + Daisy Live  
3. Set Brainflow Streamer to Network: IP 225.1.1.1, Port 6677
4. Start Session -> Start Data Stream
5. Run this script
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np


def create_simple_bar(value, width=40, min_val=-200, max_val=200):
    """Create a simple horizontal bar visualization."""
    # Normalize value to 0-1 range
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Create bar
    filled_width = int(normalized * width)
    bar = "█" * filled_width + "░" * (width - filled_width)
    return f"[{bar}] {value:6.1f}"


def main():
    print("Simple OpenBCI Stream Processor")
    print("=" * 50)
    
    # Board configuration - change this to switch board types
    BOARD_TYPE = BoardIds.CYTON_DAISY_BOARD
    
    # Setup board connection
    BoardShim.enable_dev_board_logger()
    
    params = BrainFlowInputParams()
    # These settings must match what you configure in OpenBCI GUI:
    # GUI -> System Control Panel -> Brainflow Streamer -> Network
    params.ip_port = 6677          # Port set in OpenBCI GUI
    params.ip_address = "225.1.1.1"  # Multicast IP set in OpenBCI GUI  
    params.master_board = BOARD_TYPE
    
    board = BoardShim(BoardIds.STREAMING_BOARD, params)
    
    try:
        # Connect and start streaming
        print("Connecting to OpenBCI stream...")
        board.prepare_session()
        board.start_stream()
        
        # Get board info
        sampling_rate = BoardShim.get_sampling_rate(BOARD_TYPE)
        eeg_channels = BoardShim.get_eeg_channels(BOARD_TYPE)
        
        print(f"Connected! Sampling rate: {sampling_rate} Hz")
        print(f"EEG channels: {eeg_channels}")
        print("\nStreaming data (press Ctrl+C to stop):")
        print("-" * 50)
        
        # Wait for buffer to fill
        time.sleep(2)
        
        # Stream processing loop
        while True:
            # Get recent data
            data = board.get_current_board_data(100)  # Get last 100 samples
            
            if data.size > 0:
                # Process each EEG channel
                for channel in eeg_channels:
                    if channel < data.shape[0]:
                        channel_data = data[channel]
                        if len(channel_data) > 0:
                            # Get average of recent samples
                            avg_value = np.mean(channel_data)
                            
                            # Create visual bar
                            bar = create_simple_bar(avg_value)
                            print(f"CH{channel}: {bar}")
                
                print("-" * 50)
            else:
                print("No data received...")
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        print("Stream stopped.")


if __name__ == "__main__":
    main()