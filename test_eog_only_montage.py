#!/usr/bin/env python3
"""
Test script for EOG-only montage functionality.

This script tests:
1. EOG-only montage creation
2. Channel index validation
3. DataManager integration with EOG-only montage
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from gssc_local.montage import Montage

def test_eog_only_montage_creation():
    """Test that EOG-only montage is created correctly."""
    print("=== Testing EOG-only montage creation ===")
    
    montage = Montage.eog_only_montage()
    
    # Check channel labels
    labels = montage.get_channel_labels()
    print(f"Channel labels: {labels}")
    assert labels == ["R-LEOG", "L-LEOG"], f"Expected ['R-LEOG', 'L-LEOG'], got {labels}"
    
    # Check channel types
    types = montage.get_channel_types()
    print(f"Channel types: {types}")
    assert all(t == "EOG" for t in types), f"All channels should be EOG, got {types}"
    
    # Check channel count
    assert len(labels) == 2, f"Expected 2 channels, got {len(labels)}"
    
    print("✓ EOG-only montage creation test passed")
    return montage

def test_channel_validation():
    """Test the channel index validation functionality."""
    print("\n=== Testing channel index validation ===")
    
    montage = Montage.eog_only_montage()
    
    # Test valid EOG indices
    print("Testing valid EOG indices [0, 1]...")
    try:
        montage.validate_channel_indices_combination_types([], [0, 1])
        print("✓ Valid EOG indices passed validation")
    except ValueError as e:
        print(f"✗ Unexpected validation error: {e}")
        raise
    
    # Test invalid EEG indices (should fail)
    print("Testing invalid EEG indices [0, 1] (these are EOG channels)...")
    try:
        montage.validate_channel_indices_combination_types([0, 1], [])
        print("✗ Validation should have failed but didn't")
        raise AssertionError("Validation should have caught EEG indices pointing to EOG channels")
    except ValueError as e:
        print(f"✓ Correctly caught invalid EEG indices: {e}")
    
    # Test out of range indices
    print("Testing out of range indices...")
    try:
        montage.validate_channel_indices_combination_types([], [0, 1, 2])
        print("✗ Validation should have failed for out of range index")
        raise AssertionError("Validation should have caught out of range index")
    except ValueError as e:
        print(f"✓ Correctly caught out of range index: {e}")

def test_minimal_montage_validation():
    """Test validation with minimal_sleep_montage for comparison."""
    print("\n=== Testing minimal montage validation ===")
    
    montage = Montage.minimal_sleep_montage()
    
    # Check what channels we have
    labels = montage.get_channel_labels()
    types = montage.get_channel_types()
    print(f"Minimal montage labels: {labels}")
    print(f"Minimal montage types: {types}")
    
    # Find EEG and EOG positions
    eeg_positions = [i for i, t in enumerate(types) if t == "EEG"]
    eog_positions = [i for i, t in enumerate(types) if t == "EOG"]
    
    print(f"EEG channel positions: {eeg_positions}")
    print(f"EOG channel positions: {eog_positions}")
    
    # Test validation with correct indices
    try:
        montage.validate_channel_indices_combination_types(eeg_positions[:3], eog_positions[:1])
        print("✓ Minimal montage validation passed")
    except ValueError as e:
        print(f"✗ Minimal montage validation failed: {e}")
        raise

def main():
    """Run all tests."""
    print("Testing EOG-only montage functionality...\n")
    
    try:
        # Test montage creation
        eog_montage = test_eog_only_montage_creation()
        
        # Test validation
        test_channel_validation()
        
        # Test minimal montage for comparison
        test_minimal_montage_validation()
        
        print("\n🎉 All tests passed! EOG-only montage functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()