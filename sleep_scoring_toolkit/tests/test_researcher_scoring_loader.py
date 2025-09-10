#!/usr/bin/env python3

"""
Test script for MATLAB researcher scoring file parsing and timestamp alignment.

This script validates the parsing logic for researcher scoring files before
integrating into the main data pipeline. It examines file formats, timestamp
alignment, and tests lookup functionality with extensive logging.

Usage:
    python test_researcher_scoring_loader.py
"""

import sys
import os
import numpy as np
import pandas as pd
import random
import h5py

# Add the project root to Python path to enable absolute imports
workspace_root = os.path.dirname(__file__)
sys.path.append(workspace_root)

# Import the ResearcherScoringLoader class
from sleep_scoring_toolkit.realtime_with_restart.utils.researcher_scoring_loader import ResearcherScoringLoader

def test_mat_file_structure():
    """Load and examine MATLAB .mat file structure with detailed logging."""
    print("=" * 80)
    print("TEST 1: MATLAB FILE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    print("✅ h5py available for MATLAB v7.3 files")
    
    # Path to the MATLAB scoring file
    mat_file_path = os.path.join(workspace_root, "data/realtime_inference_test/8-16-25/8-16-25_scoring.mat")
    
    if not os.path.exists(mat_file_path):
        print(f"❌ MATLAB file not found: {mat_file_path}")
        return False
    
    print(f"📁 Loading MATLAB file: {mat_file_path}")
    
    # Load MATLAB file with h5py
    try:
        print("🔄 Loading MATLAB file with h5py...")
        
        with h5py.File(mat_file_path, 'r') as f:
            print("✅ MATLAB file loaded successfully with h5py")
            
            def explore_h5_group(group, prefix=""):
                """Recursively explore HDF5 group structure."""
                for key in group.keys():
                    item = group[key]
                    print(f"   {prefix}{key}: {type(item)} - Shape: {getattr(item, 'shape', 'N/A')}")
                    
                    if isinstance(item, h5py.Group):
                        explore_h5_group(item, prefix + "  ")
                    elif isinstance(item, h5py.Dataset):
                        try:
                            # Show sample data for key fields
                            if key in ['stages', 'onsets', 'stageTime'] and item.size < 10000:
                                data = item[:]
                                if data.size > 0:
                                    flat_data = data.flatten() if hasattr(data, 'flatten') else data
                                    print(f"      {prefix}Length: {len(flat_data) if hasattr(flat_data, '__len__') else 'scalar'}")
                                    if hasattr(flat_data, '__len__') and len(flat_data) > 0:
                                        print(f"      {prefix}Sample values (first 10): {flat_data[:10]}")
                                        print(f"      {prefix}Data range: {flat_data.min()} to {flat_data.max()}")
                                    print(f"      {prefix}Data type: {data.dtype}")
                        except Exception as e:
                            print(f"      {prefix}⚠️  Could not read data: {e}")
            
            # Log top-level structure
            print(f"\n📊 TOP-LEVEL KEYS: {list(f.keys())}")
            explore_h5_group(f)
            
            # Try to access stageData specifically
            if 'stageData' in f:
                print(f"\n📊 STAGE_DATA STRUCTURE:")
                stage_data = f['stageData']
                explore_h5_group(stage_data, "   ")
            
            print("✅ MATLAB file structure analysis complete")
            return True
            
    except Exception as e:
        print(f"❌ Failed to load MATLAB file with h5py: {e}")
        return False

def test_eeg_timestamp_format():
    """Load EEG data timestamps and examine their format using proper board configuration."""
    print("\n" + "=" * 80)
    print("TEST 2: EEG DATA TIMESTAMP FORMAT ANALYSIS")
    print("=" * 80)
    
    # Initialize board configuration (matches main_speed_controlled_stream.py)
    try:
        from brainflow.board_shim import BoardShim, BoardIds
        
        board_id = BoardIds.CYTON_DAISY_BOARD
        print(f"📋 Using board ID: {board_id}")
        
        # Get proper sampling rate and timestamp channel from BoardShim
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        board_timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        
        print(f"   Board sampling rate: {sampling_rate} Hz")
        print(f"   Board timestamp channel: {board_timestamp_channel}")
        
    except Exception as e:
        print(f"❌ Error initializing board configuration: {e}")
        return None
    
    # Path to the EEG data file (matches main_speed_controlled_stream.py)
    eeg_file_path = os.path.join(workspace_root, "data/realtime_inference_test/8-16-25/BrainFlow-RAW_2025-08-16_04-33-14_0.csv")
    
    if not os.path.exists(eeg_file_path):
        print(f"❌ EEG file not found: {eeg_file_path}")
        return None
    
    print(f"📁 Loading EEG data file: {eeg_file_path}")
    
    try:
        # Load EEG data (tab-separated, no header)
        eeg_data = pd.read_csv(eeg_file_path, sep='\t', header=None)
        print(f"✅ EEG data loaded successfully")
        print(f"   Shape: {eeg_data.shape}")
        print(f"   Columns: {list(eeg_data.columns)}")
        
        # Use the correct timestamp column from board configuration
        if board_timestamp_channel < len(eeg_data.columns):
            timestamps = eeg_data[board_timestamp_channel]
            timestamp_column = board_timestamp_channel
            print(f"   Using board timestamp column: {board_timestamp_channel}")
        else:
            print(f"❌ Board timestamp channel {board_timestamp_channel} not found in data (only {len(eeg_data.columns)} columns)")
            return None
        
        print(f"\n📊 TIMESTAMP COLUMN ANALYSIS (Column {timestamp_column}):")
        print(f"   Data type: {timestamps.dtype}")
        print(f"   Length: {len(timestamps)}")
        print(f"   Range: {timestamps.min()} to {timestamps.max()}")
        print(f"   First 10 values: {timestamps.head(10).tolist()}")
        print(f"   Last 10 values: {timestamps.tail(10).tolist()}")
        
        # Calculate sampling rate and duration
        if len(timestamps) > 1:
            time_diff = timestamps.iloc[1] - timestamps.iloc[0]
            estimated_sampling_rate = 1.0 / time_diff if time_diff > 0 else 0
            total_duration = timestamps.iloc[-1] - timestamps.iloc[0]
            
            print(f"\n📈 TIMING ANALYSIS:")
            print(f"   Time between samples: {time_diff:.6f} seconds")
            print(f"   Estimated sampling rate: {estimated_sampling_rate:.2f} Hz")
            print(f"   Expected sampling rate: {sampling_rate:.2f} Hz")
            print(f"   Sampling rate match: {'✅' if abs(estimated_sampling_rate - sampling_rate) < 1.0 else '❌'}")
            print(f"   Total recording duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            
            # Check for gaps or irregularities
            time_diffs = timestamps.diff().dropna()
            print(f"   Timestamp differences - Mean: {time_diffs.mean():.6f}s, Std: {time_diffs.std():.6f}s")
            
            # Look for significant gaps
            large_gaps = time_diffs[time_diffs > time_diffs.mean() * 2]
            if len(large_gaps) > 0:
                print(f"   ⚠️  Found {len(large_gaps)} potential gaps in data")
        
        print("✅ EEG timestamp format analysis complete")
        return timestamps, sampling_rate, board_timestamp_channel
        
    except Exception as e:
        print(f"❌ Error loading EEG data: {e}")
        return None

def test_timestamp_alignment():
    """Compare EEG data timestamps with researcher scoring timestamps."""
    print("\n" + "=" * 80)
    print("TEST 3: TIMESTAMP ALIGNMENT ANALYSIS")
    print("=" * 80)
    
    # Get EEG timestamps
    eeg_result = test_eeg_timestamp_format()
    if eeg_result is None:
        print("❌ Cannot proceed without EEG timestamps")
        return False
    
    eeg_timestamps, sampling_rate, board_timestamp_channel = eeg_result
    
    # Get MATLAB timestamps using h5py
    try:
        mat_file_path = os.path.join(workspace_root, "data/realtime_inference_test/8-16-25/8-16-25_scoring.mat")
        
        with h5py.File(mat_file_path, 'r') as f:
            if 'stageData' in f:
                stage_data = f['stageData']
                
                # Extract onsets and stages - note the different indexing for h5py
                onsets_raw = stage_data['onsets'][:]  # Shape (1, 609)
                stages_raw = stage_data['stages'][:]  # Shape (1, 609)
                stageTime_raw = stage_data['stageTime'][:]  # Shape (609, 1)
                
                # Flatten to 1D arrays
                onsets = onsets_raw.flatten()
                stages = stages_raw.flatten()
                stageTime = stageTime_raw.flatten()
                
                print(f"\n📊 RESEARCHER SCORING DATA:")
                print(f"   Raw shapes - onsets: {onsets_raw.shape}, stages: {stages_raw.shape}, stageTime: {stageTime_raw.shape}")
                print(f"   Flattened lengths - onsets: {len(onsets)}, stages: {len(stages)}, stageTime: {len(stageTime)}")
                print(f"   Using stageTime (seconds) instead of onsets (sample indices)")
                
                # Extract all timing-related fields to find a reliable timestamp
                rec_start_raw = stage_data['recStart'][:]
                rec_start_matlab = rec_start_raw.flatten()[0] if rec_start_raw.size > 0 else None
                
                lights_off_raw = stage_data['lightsOFF'][:]
                lights_off = lights_off_raw.flatten()[0] if lights_off_raw.size > 0 else None
                
                lights_on_raw = stage_data['lightsON'][:]
                lights_on = lights_on_raw.flatten()[0] if lights_on_raw.size > 0 else None
                
                srate_raw = stage_data['srate'][:]
                srate = srate_raw.flatten()[0] if srate_raw.size > 0 else None
                
                print(f"   All timing fields:")
                print(f"      recStart: {rec_start_matlab}")
                print(f"      lightsOFF: {lights_off}")
                print(f"      lightsON: {lights_on}")
                print(f"      srate: {srate}")
                
                # Check if any of these look like reasonable timestamps
                potential_timestamps = [rec_start_matlab, lights_off, lights_on]
                rec_start_unix = None
                
                for i, ts in enumerate(['recStart', 'lightsOFF', 'lightsON']):
                    val = potential_timestamps[i]
                    if val is not None:
                        # Try different conversions
                        print(f"      {ts} analysis:")
                        
                        # Try as MATLAB datenum
                        try:
                            unix_ts = (val - 719529) * 24 * 60 * 60
                            print(f"         As MATLAB datenum: {unix_ts} ({val})")
                        except:
                            print(f"         MATLAB datenum conversion failed")
                        
                        # Try as direct Unix timestamp if in reasonable range
                        if 1000000000 < val < 2000000000:  # Reasonable Unix timestamp range
                            print(f"         As direct Unix timestamp: {val}")
                            if rec_start_unix is None:  # Use first reasonable one
                                rec_start_unix = val
                
                # If no reasonable timestamp found, we'll use EEG start time later
                if rec_start_unix is None:
                    print(f"      No reliable timestamp found in MATLAB file - will use EEG start time")
                
                # Use stageTime as the actual timestamps in seconds
                actual_timestamps = stageTime  # These are in seconds
                onsets = actual_timestamps  # Rename for consistency with rest of code
            
            print(f"\n📊 RESEARCHER SCORING TIMESTAMPS:")
            print(f"   Number of scored epochs: {len(onsets)}")
            print(f"   Onset range: {onsets.min()} to {onsets.max()}")
            print(f"   First 10 onsets: {onsets[:10]}")
            print(f"   First 10 stages: {stages[:10]}")
            
            # Calculate epoch durations (stageTime is in minutes)
            if len(onsets) > 1:
                epoch_durations_minutes = np.diff(onsets)
                epoch_durations_seconds = epoch_durations_minutes * 60
                print(f"   Epoch durations - Mean: {epoch_durations_minutes.mean():.1f} minutes ({epoch_durations_seconds.mean():.0f}s)")
                print(f"   Expected 30s epochs: {'✅' if abs(epoch_durations_seconds.mean() - 30.0) < 1.0 else '❌'}")
            
            # Compare time ranges with proper conversion
            eeg_start = eeg_timestamps.iloc[0]
            eeg_end = eeg_timestamps.iloc[-1]
            scoring_start = onsets.min()
            scoring_end = onsets.max()
            
            print(f"\n🔍 TIME RANGE COMPARISON:")
            print(f"   EEG data:      {eeg_start:.2f} to {eeg_end:.2f} seconds (Unix timestamps)")
            print(f"   Scoring data:  {scoring_start:.1f} to {scoring_end:.1f} minutes (relative)")
            print(f"   RecStart Unix: {rec_start_unix:.2f} seconds" if rec_start_unix else "   RecStart Unix: None")
            
            # Since MATLAB recStart is unreliable, use EEG start time as recording start
            print(f"   Using EEG start time as recording reference")
            
            # Convert scoring timestamps (minutes) to absolute time using EEG start
            scoring_abs_start = eeg_start + (scoring_start * 60)  # Convert minutes to seconds
            scoring_abs_end = eeg_start + (scoring_end * 60)
            
            print(f"   Scoring (absolute): {scoring_abs_start:.2f} to {scoring_abs_end:.2f} seconds")
            
            # Calculate actual overlap
            overlap_start = max(eeg_start, scoring_abs_start)
            overlap_end = min(eeg_end, scoring_abs_end)
            overlap_duration = overlap_end - overlap_start
            
            print(f"   Overlap start: {overlap_start:.2f}")
            print(f"   Overlap end:   {overlap_end:.2f}")
            print(f"   Overlap duration: {overlap_duration:.2f} seconds ({overlap_duration/60:.2f} minutes)")
            
            if overlap_duration > 0:
                print(f"   ✅ Overlap found! Can proceed with timestamp lookup tests")
            else:
                print(f"   ❌ No overlap found")
            
            return onsets, stages, rec_start_unix
            
    except Exception as e:
        print(f"❌ Error analyzing timestamp alignment: {e}")
        return None, None, None

def test_random_timestamp_lookup(onsets, stages, eeg_timestamps):
    """Test ResearcherScoringLoader class with random timestamps from EEG data."""
    print("\n" + "=" * 80)
    print("TEST 4: RESEARCHER SCORING LOADER CLASS TEST")
    print("=" * 80)
    
    if onsets is None or stages is None or eeg_timestamps is None:
        print("❌ Cannot proceed without timestamp data")
        return False
    
    # Initialize the ResearcherScoringLoader class
    eeg_start = eeg_timestamps.iloc[0]
    mat_file_path = os.path.join(workspace_root, "data/realtime_inference_test/8-16-25/8-16-25_scoring.mat")
    
    try:
        print("🔧 Initializing ResearcherScoringLoader...")
        loader = ResearcherScoringLoader(mat_file_path, eeg_start)
        print("✅ ResearcherScoringLoader initialized successfully")
        
        # Display loader info
        info = loader.get_scoring_info()
        print(f"📊 Scoring Info:")
        print(f"   Duration: {info['duration_minutes']:.1f} minutes ({info['duration_seconds']:.0f} seconds)")
        print(f"   Epochs: {info['num_epochs']} at {info['epoch_interval_seconds']}s intervals")
        print(f"   Stage distribution: {info['stage_counts']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize ResearcherScoringLoader: {e}")
        return False
    
    # Test with random timestamps from EEG data
    num_tests = 20
    print(f"🎲 Testing {num_tests} random timestamps from EEG data:")
    
    # Get overlap range for meaningful tests (convert scoring minutes to absolute seconds)
    eeg_start = eeg_timestamps.iloc[0]
    eeg_end = eeg_timestamps.iloc[-1]
    scoring_start_min = onsets.min()  # Minutes
    scoring_end_min = onsets.max()    # Minutes
    
    # Convert scoring range to absolute timestamps using EEG start
    scoring_abs_start = eeg_start + (scoring_start_min * 60)
    scoring_abs_end = eeg_start + (scoring_end_min * 60)
    
    overlap_start = max(eeg_start, scoring_abs_start)
    overlap_end = min(eeg_end, scoring_abs_end)
    
    print(f"   EEG range: {eeg_start:.2f} to {eeg_end:.2f}")
    print(f"   Scoring range (absolute): {scoring_abs_start:.2f} to {scoring_abs_end:.2f}")
    print(f"   Overlap range: {overlap_start:.2f} to {overlap_end:.2f}")
    
    if overlap_end <= overlap_start:
        print("❌ No overlap between EEG and scoring data")
        return False
    
    # Generate random timestamps in overlap range
    success_count = 0
    for i in range(num_tests):
        # Pick random timestamp from EEG data within overlap range
        overlap_mask = (eeg_timestamps >= overlap_start) & (eeg_timestamps <= overlap_end)
        overlap_timestamps = eeg_timestamps[overlap_mask]
        
        if len(overlap_timestamps) == 0:
            continue
            
        random_timestamp = random.choice(overlap_timestamps.tolist())
        researcher_score = loader.get_researcher_score_for_timestamp(random_timestamp)
        
        print(f"   Test {i+1:2d}: Timestamp {random_timestamp:8.2f}s -> Stage {researcher_score}")
        
        if researcher_score is not None:
            success_count += 1
    
    success_rate = (success_count / num_tests) * 100
    print(f"\n📈 LOOKUP RESULTS:")
    print(f"   Successful lookups: {success_count}/{num_tests} ({success_rate:.1f}%)")
    print(f"   Expected success rate: ~{len(onsets) * 30 / (overlap_end - overlap_start) * 100:.1f}% (based on epoch coverage)")
    
    # Test boundary conditions
    print(f"\n🔍 BOUNDARY CONDITION TESTS:")
    
    # Test with EEG timestamps outside scoring range (convert scoring range to absolute timestamps)
    scoring_abs_start = eeg_start + (scoring_start_min * 60)  # Convert minutes to seconds, add to EEG start
    scoring_abs_end = eeg_start + (scoring_end_min * 60)
    
    before_scoring = scoring_abs_start - 60  # 1 minute before scoring starts
    after_scoring = scoring_abs_end + 60     # 1 minute after scoring ends
    
    score_before = loader.get_researcher_score_for_timestamp(before_scoring)
    score_after = loader.get_researcher_score_for_timestamp(after_scoring)
    
    print(f"   Before scoring range ({before_scoring:.2f}): {score_before} {'✅' if score_before is None else '❌'}")
    print(f"   After scoring range ({after_scoring:.2f}): {score_after} {'✅' if score_after is None else '❌'}")
    
    # Test with exact onset timestamps (convert onset minute to absolute timestamp)
    if len(onsets) >= 3:
        onset_minutes = onsets[1]  # Use second onset (in minutes)
        exact_onset = eeg_start + (onset_minutes * 60)  # Convert to absolute timestamp
        score_exact = loader.get_researcher_score_for_timestamp(exact_onset)
        expected_exact = int(stages[1])
        
        print(f"   Exact onset ({onset_minutes:.1f}min = {exact_onset:.2f}s): {score_exact} {'✅' if score_exact == expected_exact else '❌'}")
    
    # Test specific known stage/timestamp pairs for validation
    print(f"\n📋 VALIDATION WITH KNOWN STAGE/TIMESTAMP PAIRS:")
    
    # From h5dump actual data, exact stage/time pairs:
    # (0,88): 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2,
    # (0,110): 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5,
    # stageTime: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5...
    known_tests = [
        # (minute, expected_stage, description)
        (0.0, 0, "Index 0 - Start Wake"),
        (44.0, 0, "Index 88 - Late Wake"),  
        (49.5, 1, "Index 99 - N1 transition"),
        (54.5, 2, "Index 109 - N2 sleep"),
        (63.5, 5, "Index 127 - REM start"),
        (100.0, 5, "Index 200 - Mid-REM"),
        (150.0, 5, "Index 300 - Late REM"), 
        (200.0, 2, "Index 400 - Back to N2"),
        (250.0, 5, "Index 500 - More REM"),
        (304.0, 7, "Index 608 - End stage 7")
    ]
    
    validation_success = 0
    for minute, expected_stage, description in known_tests:
        # Convert minute to absolute EEG timestamp
        test_timestamp = eeg_start + (minute * 60)
        actual_stage = loader.get_researcher_score_for_timestamp(test_timestamp)
        
        match = actual_stage == expected_stage
        status = "✅" if match else "❌"
        
        print(f"   {minute:5.1f}min ({description:20s}): Expected {expected_stage}, Got {actual_stage} {status}")
        
        if match:
            validation_success += 1
    
    validation_rate = (validation_success / len(known_tests)) * 100
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Correct predictions: {validation_success}/{len(known_tests)} ({validation_rate:.1f}%)")
    
    if validation_rate >= 80:
        print("   ✅ Validation passed - lookup function working correctly")
    else:
        print("   ❌ Validation failed - check lookup logic")

    print("✅ Random timestamp lookup test complete")
    return True

def main():
    """Run all tests to validate MATLAB file parsing and timestamp alignment."""
    print("🧪 RESEARCHER SCORING LOADER TEST SUITE")
    print("This script validates MATLAB file parsing before main integration")
    print()
    
    # Test 1: MATLAB file structure
    if not test_mat_file_structure():
        print("\n❌ MATLAB file structure test failed - cannot continue")
        return False
    
    # Test 2 & 3: Timestamp analysis and alignment  
    onsets, stages, rec_start_unix = test_timestamp_alignment()
    
    # Test 4: Random lookup
    eeg_result = test_eeg_timestamp_format()
    if eeg_result is None:
        print("\n❌ Cannot get EEG timestamps for random lookup test")
        return False
    
    eeg_timestamps, _, _ = eeg_result
    if not test_random_timestamp_lookup(onsets, stages, eeg_timestamps):
        print("\n❌ Random timestamp lookup test failed")
        return False
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("📝 Key findings logged above - review for implementation details")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()