#!/usr/bin/env python3
"""
Test script to explore what values can be inserted into BrainFlow marker channel.
Tests various data types and values to understand the limitations.
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

def test_marker_values():
    """Test inserting different types of values into the marker channel."""
    
    print("Testing different marker values with BrainFlow...")
    print("=" * 50)
    
    # Initialize synthetic board for safe testing
    board_id = BoardIds.SYNTHETIC_BOARD
    params = BrainFlowInputParams()
    board = BoardShim(board_id, params)
    
    try:
        board.prepare_session()
        board.start_stream()
        
        # Test different marker values
        test_values = [
            ("Integer positive", 1),
            ("Integer large", 9999),
            ("Float positive", 3.14159),
            ("Float negative", -2.718),
            ("Small decimal", 0.001),
            ("Scientific notation", 1e6),
            ("Negative integer", -100),
            ("Very large number", 1234567890.123456),
        ]
        
        print("Inserting different marker values:")
        print("-" * 40)
        
        successful_markers = []
        
        for description, value in test_values:
            try:
                board.insert_marker(value)
                print(f"✓ {description:20}: {value}")
                successful_markers.append((description, value))
                time.sleep(0.5)  # Small delay between markers
            except Exception as e:
                print(f"✗ {description:20}: {value} - Error: {e}")
        
        # Test zero value (should not work as it's reserved)
        print("\nTesting reserved values:")
        print("-" * 40)
        try:
            board.insert_marker(0)
            print("✓ Zero value: 0 (unexpectedly worked)")
        except Exception as e:
            print(f"✗ Zero value: 0 - Error: {e}")
        
        # Test invalid types
        print("\nTesting invalid data types:")
        print("-" * 40)
        
        invalid_values = [
            ("String", "test"),
            ("List", [1, 2, 3]),
            ("None", None),
            ("Boolean True", True),
            ("Boolean False", False),
        ]
        
        for description, value in invalid_values:
            try:
                board.insert_marker(value)
                print(f"✓ {description:20}: {value} (unexpectedly worked)")
            except Exception as e:
                print(f"✗ {description:20}: {value} - Error: {type(e).__name__}")
        
        # Collect data and verify markers
        print("\n" + "=" * 50)
        print("Retrieving and verifying inserted markers:")
        print("-" * 40)
        
        time.sleep(1)  # Allow more data collection
        data = board.get_board_data()
        
        marker_channel = BoardShim.get_marker_channel(board_id)
        timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        
        if data.size > 0:
            marker_data = data[marker_channel]
            timestamp_data = data[timestamp_channel]
            
            # Find all non-zero markers
            marker_indices = [i for i, x in enumerate(marker_data) if x != 0]
            
            print(f"Found {len(marker_indices)} markers in the data:")
            
            for i, idx in enumerate(marker_indices):
                marker_value = marker_data[idx]
                timestamp = timestamp_data[idx]
                
                # Try to match with our test values
                matched_description = "Unknown"
                for desc, val in successful_markers:
                    if abs(float(val) - marker_value) < 1e-10:  # Account for floating point precision
                        matched_description = desc
                        break
                
                print(f"  {i+1}. Value: {marker_value:15.6f} | Timestamp: {timestamp:.6f} | Type: {matched_description}")
        else:
            print("No data collected")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except:
            pass

if __name__ == "__main__":
    test_marker_values()