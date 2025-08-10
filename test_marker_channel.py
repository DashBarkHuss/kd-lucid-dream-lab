#!/usr/bin/env python3
"""
Simple script to test BrainFlow marker channel functionality.
This script demonstrates how to get the marker channel index for different board types.
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def test_marker_channel():
    """Test getting marker channel for different board types."""
    
    # Test with different board types commonly used
    board_types = [
        (BoardIds.GANGLION_BOARD, "Ganglion Board"),
        (BoardIds.CYTON_BOARD, "Cyton Board"),
        (BoardIds.CYTON_DAISY_BOARD, "Cyton Daisy Board"),
        (BoardIds.SYNTHETIC_BOARD, "Synthetic Board"),
    ]
    
    print("Testing marker channel availability for different BrainFlow boards:")
    print("-" * 60)
    
    for board_id, board_name in board_types:
        try:
            marker_channel = BoardShim.get_marker_channel(board_id)
            print(f"{board_name:20} | Marker Channel: {marker_channel}")
        except Exception as e:
            print(f"{board_name:20} | Error: {e}")
    
    print("-" * 60)
    
    # Test with synthetic board to show actual usage
    print("\nTesting with synthetic board (safe for testing):")
    
    try:
        board_id = BoardIds.SYNTHETIC_BOARD
        params = BrainFlowInputParams()
        
        board = BoardShim(board_id, params)
        board.prepare_session()
        
        # Get channel info
        marker_channel = BoardShim.get_marker_channel(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        
        print(f"Board ID: {board_id}")
        print(f"Marker channel index: {marker_channel}")
        print(f"EEG channels: {eeg_channels}")
        print(f"Timestamp channel: {timestamp_channel}")
        
        # Start streaming and insert a marker
        board.start_stream()
        
        # Insert a test marker
        test_marker_value = 123
        board.insert_marker(test_marker_value)
        print(f"Inserted marker with value: {test_marker_value}")
        
        # Get some data
        import time
        time.sleep(1)  # Collect data for 1 second
        
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()
        
        if data.size > 0:
            marker_data = data[marker_channel]
            # Find non-zero markers
            marker_indices = [i for i, x in enumerate(marker_data) if x != 0]
            
            print(f"\nData shape: {data.shape}")
            print(f"Marker channel data (first 10 samples): {marker_data[:10]}")
            print(f"Found {len(marker_indices)} markers at indices: {marker_indices}")
            
            if marker_indices:
                for idx in marker_indices:
                    timestamp = data[timestamp_channel][idx]
                    marker_value = marker_data[idx]
                    print(f"Marker {marker_value} at timestamp {timestamp}")
        else:
            print("No data collected")
            
    except Exception as e:
        print(f"Error testing synthetic board: {e}")
        try:
            board.release_session()
        except:
            pass

if __name__ == "__main__":
    test_marker_channel()