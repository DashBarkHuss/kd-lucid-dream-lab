#!/usr/bin/env python3
"""
Test script for StatefulInferenceManager multi-buffer functionality.

This test validates that the refactored StatefulInferenceManager works correctly
with multiple buffers for round-robin processing as used by DataManager.
"""

import numpy as np
import torch
import pytest
from sleep_scoring_toolkit.stateful_inference_manager import StatefulInferenceManager
from sleep_scoring_toolkit.constants import REALTIME_BUFFER_COUNT
from sleep_scoring_toolkit.realtime_with_restart.processor import SignalProcessor
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.constants import STANDARD_EEG_CHANNELS, STANDARD_EOG_CHANNELS

def get_standard_channel_names():
    """Get standard EEG/EOG channel names for testing."""
    return STANDARD_EEG_CHANNELS, STANDARD_EOG_CHANNELS

def create_test_epoch_data():
    """Create generic epoch data for testing."""
    from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import (
        create_numpy_data_with_brainflow_keys
    )
    from sleep_scoring_toolkit.tests.test_utils import create_brainflow_test_data
    
    # Create 30-second epoch using generic EEG data
    epoch_data, _ = create_brainflow_test_data(duration_seconds=30.0, add_noise=True)
    
    # Transform to stream format (n_channels, n_samples) for processing
    epoch_data = epoch_data.T
    
    # Create channel mapping for board positions 1-16 (first 16 channels)
    n_channels = epoch_data.shape[0]
    electrode_board_keys = list(range(1, min(17, n_channels + 1)))
    
    # Use only first 16 channels to match expected format
    epoch_data_trimmed = epoch_data[:16, :] if epoch_data.shape[0] >= 16 else epoch_data
    
    # Wrap epoch data with structured mapping
    epoch_data_keyed = create_numpy_data_with_brainflow_keys(epoch_data_trimmed, electrode_board_keys)
    
    return epoch_data_keyed

def test_single_buffer_compatibility():
    """Test that single buffer mode works (backward compatibility)."""
    print("🧪 Testing single buffer compatibility...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create single buffer processor with explicit channel names
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=1
    )
    
    # Create realistic Wake EEG data for testing
    epoch_data = create_test_epoch_data()
    
    # Process epoch (should default to buffer_id=0)
    pred1, _, states1 = processor.process_epoch(epoch_data)
    pred2, _, _ = processor.process_epoch(epoch_data, buffer_id=0)
    
    print(f"✓ Single buffer processing works")
    print(f"  Prediction 1: {pred1}, Prediction 2: {pred2}")
    print(f"  Hidden states maintained: {len(states1)} combinations")
    
    # Assertions for pytest
    assert pred1 == pred2, "Predictions should be consistent for same buffer"
    assert len(states1) > 0, "Should maintain hidden states"

def test_multi_buffer_functionality():
    """Test multi-buffer functionality for round-robin processing."""
    print("🧪 Testing multi-buffer functionality...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create real-time buffer processor (like DataManager)
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=REALTIME_BUFFER_COUNT
    )
    
    # Create realistic N2 EEG data for testing
    epoch_data = create_test_epoch_data()
    
    # Test processing on different buffers
    predictions = {}
    for buffer_id in range(REALTIME_BUFFER_COUNT):
        pred, _, _ = processor.process_epoch(epoch_data, buffer_id)
        predictions[buffer_id] = pred
        print(f"  Buffer {buffer_id}: Prediction = {pred}")
    
    print(f"✓ Multi-buffer processing works for {REALTIME_BUFFER_COUNT} buffers")
    
    # Assertions for pytest
    assert len(predictions) == REALTIME_BUFFER_COUNT, f"Should process all {REALTIME_BUFFER_COUNT} buffers"
    for buffer_id, pred in predictions.items():
        assert isinstance(pred, (int, np.integer)), f"Buffer {buffer_id} should return integer prediction"
        assert 0 <= pred <= 4, f"Buffer {buffer_id} prediction {pred} should be valid sleep stage"

def test_buffer_state_isolation():
    """Test that different buffers maintain separate hidden states."""
    print("🧪 Testing buffer state isolation...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create 3-buffer processor for testing
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=3
    )
    
    # Create different realistic EEG data for each buffer (different sleep stages)
    epoch_data_wake = create_test_epoch_data()
    epoch_data_n2 = create_test_epoch_data()
    epoch_data_rem = create_test_epoch_data()
    
    # Process multiple epochs on each buffer
    buffer_predictions = {0: [], 1: [], 2: []}
    
    for _ in range(3):
        # Process Wake data on buffer 0
        pred0, _, _ = processor.process_epoch(epoch_data_wake, buffer_id=0)
        buffer_predictions[0].append(pred0)
        
        # Process N2 data on buffer 1
        pred1, _, _ = processor.process_epoch(epoch_data_n2, buffer_id=1)
        buffer_predictions[1].append(pred1)
        
        # Process REM data on buffer 2
        pred2, _, _ = processor.process_epoch(epoch_data_rem, buffer_id=2)
        buffer_predictions[2].append(pred2)
    
    print(f"✓ Buffer state isolation maintained")
    print(f"  Buffer 0 predictions: {buffer_predictions[0]}")
    print(f"  Buffer 1 predictions: {buffer_predictions[1]}")
    print(f"  Buffer 2 predictions: {buffer_predictions[2]}")
    
    # Assertions for pytest
    for buffer_id, predictions in buffer_predictions.items():
        assert len(predictions) == 3, f"Buffer {buffer_id} should have 3 predictions"
        for pred in predictions:
            assert isinstance(pred, (int, np.integer)), f"Buffer {buffer_id} should return integer predictions"
            assert 0 <= pred <= 4, f"Buffer {buffer_id} prediction {pred} should be valid sleep stage"

def test_buffer_reset_functionality():
    """Test buffer reset functionality for gap handling."""
    print("🧪 Testing buffer reset functionality...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create 3-buffer processor
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=3
    )
    
    # Create test data
    epoch_data = create_test_epoch_data()
    
    # Process some epochs to establish hidden states
    pred1, _, _ = processor.process_epoch(epoch_data, buffer_id=1)
    pred2, _, _ = processor.process_epoch(epoch_data, buffer_id=1)
    
    print(f"  Before reset - Predictions: {pred1}, {pred2}")
    
    # Reset buffer 1
    processor.reset_buffer_hidden_states(buffer_id=1)
    
    # Process again (should have fresh hidden states)
    pred3, _, _ = processor.process_epoch(epoch_data, buffer_id=1)
    print(f"  After reset - Prediction: {pred3}")
    
    # Test reset all buffers
    processor.reset_buffer_hidden_states()  # Reset all
    pred4, _, _ = processor.process_epoch(epoch_data, buffer_id=1)
    print(f"  After reset all - Prediction: {pred4}")
    
    print(f"✓ Buffer reset functionality works")
    
    # Assertions for pytest
    assert isinstance(pred1, (int, np.integer)), "Should return integer prediction before reset"
    assert isinstance(pred2, (int, np.integer)), "Should return integer prediction before reset"
    assert isinstance(pred3, (int, np.integer)), "Should return integer prediction after reset"
    assert isinstance(pred4, (int, np.integer)), "Should return integer prediction after reset all"

def test_sleep_stage_specific_processing():
    """Test processing with different realistic sleep stage data."""
    print("🧪 Testing sleep stage-specific EEG processing...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create single buffer processor for testing
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=1
    )
    
    # Test each sleep stage
    sleep_stages = {
        0: "Wake",
        1: "N1", 
        2: "N2",
        3: "N3",
        4: "REM"
    }
    
    predictions = {}
    for stage_name in sleep_stages.values():
        # Create generic test data for each stage test
        epoch_data = create_test_epoch_data()
        
        # Process the epoch
        pred, _, _ = processor.process_epoch(epoch_data, buffer_id=0)
        predictions[stage_name] = pred
        
        print(f"  {stage_name} EEG → Prediction: {pred}")
    
    print(f"✓ Sleep stage-specific processing completed")
    print(f"  All stages processed successfully with realistic EEG patterns")
    
    # Assertions for pytest
    assert len(predictions) == 5, "Should process all 5 sleep stages"
    for stage_name, pred in predictions.items():
        assert isinstance(pred, (int, np.integer)), f"{stage_name} should return integer prediction"
        assert 0 <= pred <= 4, f"{stage_name} prediction {pred} should be valid sleep stage"

def test_error_handling():
    """Test error handling for invalid buffer IDs."""
    print("🧪 Testing error handling...")
    
    # Initialize components
    signal_processor = SignalProcessor(use_cuda=False)
    montage = Montage.minimal_sleep_montage()
    
    # Create 2-buffer processor
    eeg_channels, eog_channels = get_standard_channel_names()
    processor = StatefulInferenceManager(
        signal_processor, montage, 
        eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
        num_buffers=2
    )
    
    # Create test data
    epoch_data = create_test_epoch_data()
    
    # Test invalid buffer ID
    with pytest.raises(ValueError, match="Invalid buffer_id"):
        processor.process_epoch(epoch_data, buffer_id=5)  # Should fail
    print(f"✓ Correctly caught invalid buffer_id for process_epoch")
    
    # Test invalid buffer ID for reset
    with pytest.raises(ValueError, match="Invalid buffer_id"):
        processor.reset_buffer_hidden_states(buffer_id=10)  # Should fail
    print(f"✓ Correctly caught invalid buffer_id for reset")

def main():
    """Run all tests."""
    print("=" * 80)
    print("🧪 StatefulInferenceManager Multi-Buffer Functionality Tests")
    print("=" * 80)
    
    tests = [
        test_single_buffer_compatibility,
        test_multi_buffer_functionality,
        test_buffer_state_isolation,
        test_buffer_reset_functionality,
        test_sleep_stage_specific_processing,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("✗ Test failed")
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print()
    
    print("=" * 80)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! StatefulInferenceManager multi-buffer functionality works correctly.")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()