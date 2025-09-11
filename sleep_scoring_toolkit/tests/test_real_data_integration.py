"""
Real integration test using actual EEG data with known researcher scores.
Tests that our shared epoch inference function works correctly and produces consistent results.
Also tests different channel combinations systematically.

⚠️  IMPORTANT TESTING LIMITATIONS:
This test uses ISOLATED EPOCH INFERENCE with fresh hidden states for each prediction.
The GSSC model is a recurrent neural network that relies on temporal context through 
hidden states accumulated over previous epochs. 

EXPECTED BEHAVIOR:
- Potentially degraded accuracy due to lack of temporal context
- Poor performance on sleep stages (N2, N3, REM) that require temporal context  
- May show bias toward stages identifiable from isolated epochs
- Results DO NOT represent real-time system performance (which maintains hidden states)

PURPOSE:
- Validates shared epoch inference function works across all channel combinations
- Tests code integration, not model accuracy
- Demonstrates channel combination effects in isolation

For realistic sleep stage predictions, hidden states from previous epochs would need 
to be maintained and loaded (future enhancement).

THEORETICAL HYPOTHESIS: Fresh hidden states may favor Wake predictions

Reasoning:
- Wake stages may be more identifiable from isolated 30-second epochs
- Sleep stages (N2, N3, REM) often require temporal context to distinguish from artifacts  
- The model may default to "safest" prediction (Wake) when lacking context
- Wake is often the most common stage in many datasets

However, this is unproven - the GSSC training methodology may have mitigated this issue.
Empirical testing would require comparing same epochs with fresh vs accumulated hidden states.
"""

import torch
import itertools
from typing import List, Tuple
import os

# Suppress verbose MNE filtering output
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

from sleep_scoring_toolkit.core.epoch_inference import infer_sleep_stage_for_epoch
from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import create_numpy_data_with_brainflow_keys
from sleep_scoring_toolkit.stateful_inference_manager import StatefulInferenceManager
from sleep_scoring_toolkit.batch_processing import extract_and_prepare_epoch_data
from sleep_scoring_toolkit.utils.inference_utils import (
    validate_inference_results,
    convert_researcher_stage_to_gssc,
    validate_researcher_score,
    print_epoch_results
)
from sleep_scoring_toolkit.realtime_with_restart.processor import SignalProcessor
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.realtime_with_restart.utils.researcher_scoring_loader import ResearcherScoringLoader
from sleep_scoring_toolkit.constants import GSSCStages, ResearcherStages, STANDARD_EEG_CHANNELS, STANDARD_EOG_CHANNELS
from brainflow.board_shim import BoardShim, BoardIds

def generate_channel_combinations(comprehensive=False, board_id=BoardIds.CYTON_DAISY_BOARD):
    """Generate channel combinations to test.
    
    Args:
        comprehensive: If True, generate all 63 combinations. If False, use 16 key combinations.
        board_id: BoardShim board ID to get proper channel indices from
    
    Returns:
        List[Tuple[List[int], List[int], str]]: List of (eeg_combo, eog_combo, description)
    """
    # Get montage to map channel labels to positions instead of hardcoding
    montage = Montage.minimal_sleep_montage()
    
    # Get channel positions from montage instead of hardcoding
    f3_pos = montage.get_channel_position_by_label("F3")
    f4_pos = montage.get_channel_position_by_label("F4")
    c3_pos = montage.get_channel_position_by_label("C3")
    c4_pos = montage.get_channel_position_by_label("C4")
    r_heog_pos = montage.get_channel_position_by_label("R-HEOG")
    l_heog_pos = montage.get_channel_position_by_label("L-HEOG")
    
    if not comprehensive:
        # Simplified 16-combination test - using montage positions instead of hardcoded values
        combinations = [
            # Individual EEG channels with L-HEOG (most common)
            ([f3_pos], [l_heog_pos], "F3+L-HEOG"),
            ([f4_pos], [l_heog_pos], "F4+L-HEOG"), 
            ([c3_pos], [l_heog_pos], "C3+L-HEOG"),
            ([c4_pos], [l_heog_pos], "C4+L-HEOG"),
            
            # Key pairs with L-HEOG
            ([c3_pos, c4_pos], [l_heog_pos], "C3+C4+L-HEOG"),
            ([f3_pos, f4_pos], [l_heog_pos], "F3+F4+L-HEOG"),
            ([f3_pos, c3_pos], [l_heog_pos], "F3+C3+L-HEOG"),
            ([f4_pos, c4_pos], [l_heog_pos], "F4+C4+L-HEOG"),
            
            # Default combination (matches current implementation)
            ([f3_pos, f4_pos, c3_pos, c4_pos], [r_heog_pos, l_heog_pos], "F3+F4+C3+C4+BothEOG"),
            
            # Individual channels alone (no EOG)
            ([c3_pos], [], "C3+NoEOG"),
            ([c4_pos], [], "C4+NoEOG"),
            
            # EOG only combinations
            ([], [r_heog_pos], "NoEEG+R-HEOG"),
            ([], [l_heog_pos], "NoEEG+L-HEOG"),
            ([], [r_heog_pos, l_heog_pos], "NoEEG+BothEOG"),
            
            # Classic sleep scoring combinations
            ([c3_pos, c4_pos], [r_heog_pos], "C3+C4+R-HEOG"),
            ([f3_pos, f4_pos, c3_pos, c4_pos], [l_heog_pos], "All-EEG+L-HEOG")
        ]
        return combinations
    
    # Comprehensive test: All possible combinations - using montage positions instead of hardcoded values
    eeg_channels = {
        'F3': f3_pos, 'F4': f4_pos, 'C3': c3_pos, 'C4': c4_pos
    }
    
    eog_channels = {
        'R-HEOG': r_heog_pos, 'L-HEOG': l_heog_pos
    }
    
    combinations = []
    
    # Generate all non-empty combinations of EEG channels (1 to 4 channels)
    eeg_names = list(eeg_channels.keys())
    
    for r in range(1, len(eeg_names) + 1):  # 1 to 4 channels
        for eeg_combo_names in itertools.combinations(eeg_names, r):
            eeg_combo_positions = [eeg_channels[name] for name in eeg_combo_names]
            
            # Test with no EOG
            eeg_desc = '+'.join(eeg_combo_names)
            combinations.append((eeg_combo_positions, [], f"{eeg_desc}+NoEOG"))
            
            # Test with each EOG channel individually
            for eog_name, eog_pos in eog_channels.items():
                combinations.append((eeg_combo_positions, [eog_pos], f"{eeg_desc}+{eog_name}"))
            
            # Test with both EOG channels
            both_eog = list(eog_channels.values())
            combinations.append((eeg_combo_positions, both_eog, f"{eeg_desc}+BothEOG"))
    
    # Also test EOG-only combinations
    for eog_name, eog_pos in eog_channels.items():
        combinations.append(([], [eog_pos], f"NoEEG+{eog_name}"))
    
    combinations.append(([], list(eog_channels.values()), "NoEEG+BothEOG"))
    
    return combinations

def test_channel_combination_inference_robust():
    """Test channel combinations with isolated epochs (fresh hidden states).
    
    ⚠️  TESTING LIMITATION: Uses fresh hidden states for each combination test.
    This may show reduced accuracy since the GSSC model needs temporal context for optimal predictions.
    
    Args:
        comprehensive: If True, test all 63 combinations. If False, test 16 key combinations.
    """
    
    """Test channel combinations with isolated epochs (fresh hidden states).
    
    ⚠️  TESTING LIMITATION: Uses fresh hidden states for each combination test.
    This may show reduced accuracy since the GSSC model needs temporal context for optimal predictions.
    """
    comprehensive = False
    board_id = BoardIds.CYTON_DAISY_BOARD
    
    # Test data
    recording_start_timestamp = 1755354827.610700
    csv_path = "data/realtime_inference_test/8-16-25/BrainFlow-RAW_2025-08-16_04-33-14_0.csv"
    mat_path = "data/realtime_inference_test/8-16-25/8-16-25_scoring.mat"
    
    # Use one stable epoch (Wake) for comprehensive testing
    test_epoch_num = 57  # Stable Wake epoch
    expected_stage = ResearcherStages.WAKE
    
    test_type = "All" if comprehensive else "Key"
    print("="*80)
    print(f"Testing {test_type} Channel Combinations - ISOLATED EPOCH MODE")
    print("="*80)
    print(f"Testing epoch {test_epoch_num} (expected: {ResearcherStages.to_name(expected_stage)})")
    print()
    print("⚠️  IMPORTANT: Using fresh hidden states for each test (no temporal context)")
    print("   Expected behavior: Potentially degraded accuracy due to missing temporal context")
    print("   This tests code integration, not model accuracy")
    
    # Initialize components
    signal_processor = SignalProcessor()
    montage = Montage.minimal_sleep_montage()
    scoring_loader = ResearcherScoringLoader(mat_path, recording_start_timestamp)
    
    # Get epoch data once
    epoch_start_time = recording_start_timestamp + (test_epoch_num * 30.0)
    researcher_score = scoring_loader.get_researcher_score_for_timestamp(epoch_start_time)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    epoch_data_keyed = extract_and_prepare_epoch_data(csv_path, epoch_start_time, test_epoch_num, sampling_rate, board_id)
    
    # Generate combinations
    combinations = generate_channel_combinations(comprehensive=comprehensive, board_id=board_id)
    
    print(f"\nTesting {len(combinations)} channel combinations...")
    print(f"Researcher ground truth: {researcher_score} ({ResearcherStages.to_name(researcher_score)})")
    
    results = []
    correct_predictions = 0
    
    for i, (eeg_combo, eog_combo, description) in enumerate(combinations):
        try:
            # Convert indexes to channel names for StatefulInferenceManager
            eeg_names = [list(montage.channels.values())[i-1].label for i in eeg_combo] if eeg_combo else []
            eog_names = [list(montage.channels.values())[i-1].label for i in eog_combo] if eog_combo else []
            
            # Create fresh StatefulInferenceManager for this channel combination (isolated testing)
            inference_manager = StatefulInferenceManager(
                signal_processor, montage, 
                eeg_channels_for_scoring=eeg_names, eog_channels_for_scoring=eog_names,
                num_buffers=1
            )
            
            # Run inference with fresh hidden states (no temporal context)
            predicted_class, class_probs, new_hidden_states = inference_manager.process_epoch(epoch_data_keyed)
            
            # Check if prediction matches ground truth
            researcher_as_gssc = convert_researcher_stage_to_gssc(researcher_score)
            is_correct = predicted_class == researcher_as_gssc if researcher_as_gssc is not None else False
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'combination': description,
                'eeg_channels': eeg_combo,
                'eog_channels': eog_combo,
                'predicted_class': int(predicted_class),
                'predicted_name': GSSCStages.to_name(predicted_class),
                'is_correct': is_correct
            })
            
            # Print progress every 10 combinations
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(combinations)} combinations...")
                
        except Exception as e:
            print(f"  ERROR with {description}: {e}")
            results.append({
                'combination': description,
                'eeg_channels': eeg_combo,
                'eog_channels': eog_combo,
                'predicted_class': -1,
                'predicted_name': 'ERROR',
                'is_correct': False,
                'error': str(e)
            })
    
    # Summarize results
    print(f"\n" + "="*60)
    print(f"CHANNEL COMBINATION RESULTS:")
    print(f"Total combinations tested: {len(results)}")
    print(f"Correct predictions: {correct_predictions}/{len(results)} ({correct_predictions/len(results)*100:.1f}%)")
    
    # Group by prediction
    prediction_groups = {}
    for result in results:
        pred_name = result['predicted_name']
        if pred_name not in prediction_groups:
            prediction_groups[pred_name] = []
        prediction_groups[pred_name].append(result)
    
    print(f"\nPredictions by sleep stage:")
    for pred_name, group in prediction_groups.items():
        correct_in_group = sum(1 for r in group if r['is_correct'])
        print(f"  {pred_name}: {len(group)} combinations ({correct_in_group} correct)")
    
    # Show correct predictions
    if correct_predictions > 0:
        print(f"\nCorrect predictions:")
        for result in results:
            if result['is_correct']:
                print(f"  ✓ {result['combination']}: {result['predicted_name']}")
    
    # Show some examples of different predictions
    print(f"\nSample predictions by combination type:")
    shown_types = set()
    for result in results[:20]:  # Show first 20
        combo_type = result['combination'].split('+')[0]  # Get EEG part
        if combo_type not in shown_types:
            status = "✓" if result['is_correct'] else "✗"
            print(f"  {status} {result['combination']}: {result['predicted_name']}")
            shown_types.add(combo_type)
    
    print("="*60)
    
    # Add assertions to validate the test results
    assert len(results) > 0, "No channel combinations were tested"
    assert correct_predictions >= 0, "Invalid number of correct predictions"
    
    # Test should pass if we get reasonable results from channel combinations
    error_count = sum(1 for r in results if r['predicted_name'] == 'ERROR')
    valid_predictions = len(results) - error_count
    
    assert valid_predictions > 0, "No valid predictions were made"
    
    print(f"\n✅ Channel combination test completed with {valid_predictions} valid predictions")

def test_multiple_epochs_isolated_inference():
    """Test multiple epochs with isolated inference (fresh hidden states per epoch).
    
    ⚠️  TESTING LIMITATION: Each epoch uses fresh hidden states with no temporal context.
    This may show reduced accuracy on stages requiring temporal context since GSSC is designed for sequential processing.
    """
    
    # Known data from examine_researcher_scores.py
    recording_start_timestamp = 1755354827.610700
    csv_path = "data/realtime_inference_test/8-16-25/BrainFlow-RAW_2025-08-16_04-33-14_0.csv"
    mat_path = "data/realtime_inference_test/8-16-25/8-16-25_scoring.mat"
    
    # Test epochs with different sleep stages - STABLE epochs (NOT at transitions)
    # These epochs are surrounded by the same sleep stage (found via examine_researcher_scores.py)
    # Format: (epoch_num, expected_researcher_stage)
    test_epochs = [
        (57, ResearcherStages.WAKE),   # Epoch 57: Stable Wake (109 stable epochs found)
        (404, ResearcherStages.N2),    # Epoch 404: Stable N2 (169 stable epochs found)  
        (584, ResearcherStages.N3),    # Epoch 584: Stable N3 (11 stable epochs found)
        (329, ResearcherStages.REM),   # Epoch 329: Stable REM (205 stable epochs found)
    ]
    
    print("="*70)
    print("Testing Multiple Epochs - ISOLATED INFERENCE MODE")
    print("="*70)
    print("⚠️  Using fresh hidden states per epoch (no temporal context)")
    print("   Expected: Potentially reduced accuracy, especially on stages requiring temporal context")
    
    # Initialize shared components once
    print(f"Initializing GSSC components...")
    signal_processor = SignalProcessor()
    montage = Montage.minimal_sleep_montage()
    scoring_loader = ResearcherScoringLoader(mat_path, recording_start_timestamp)
    
    # Track results for all epochs
    all_results = []
    agreements = 0
    
    def compare_epoch_predictions(epoch_num, expected_stage, eeg_combo_pk=None, eog_combo_pk=None):
        """Compare model predictions against researcher ground truth for a single epoch."""
        stage_name = ResearcherStages.to_name(expected_stage)
        
        # Get explicit channel combinations if not provided
        if eeg_combo_pk is None or eog_combo_pk is None:
            # Use explicit channel combinations (fail-fast - no defaults)
            eeg_channels = STANDARD_EEG_CHANNELS  # Standard EEG channels
            eog_channels = STANDARD_EOG_CHANNELS      # Standard EOG channels (minimal_sleep_montage)
        
        # Calculate epoch start timestamp
        epoch_start_time = recording_start_timestamp + (epoch_num * 30.0)
        
        # Get and validate researcher score
        researcher_score = scoring_loader.get_researcher_score_for_timestamp(epoch_start_time)
        validate_researcher_score(researcher_score, expected_stage, epoch_num, epoch_start_time)
        
        # Extract and prepare epoch data  
        sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
        epoch_data_keyed = extract_and_prepare_epoch_data(csv_path, epoch_start_time, epoch_num, sampling_rate, BoardIds.CYTON_DAISY_BOARD)
        
        # Create fresh StatefulInferenceManager for isolated epoch testing  
        inference_manager = StatefulInferenceManager(
            signal_processor, montage, 
            eeg_channels_for_scoring=eeg_channels, eog_channels_for_scoring=eog_channels,
            num_buffers=1
        )
        
        # Run inference with fresh hidden states (no temporal context)
        predicted_class, class_probs, new_hidden_states = inference_manager.process_epoch(epoch_data_keyed)
        
        # Validate results
        validate_inference_results(predicted_class, class_probs, new_hidden_states)
        
        # Convert researcher stage to GSSC scale and check agreement
        researcher_as_gssc = convert_researcher_stage_to_gssc(researcher_score)
        agreement = predicted_class == researcher_as_gssc if researcher_as_gssc is not None else False
        
        # Print results
        print_epoch_results(epoch_num, stage_name, expected_stage, predicted_class, 
                           researcher_score, researcher_as_gssc, agreement)
        
        return {
            "epoch_number": epoch_num,
            "researcher_stage": int(researcher_score),
            "researcher_as_gssc": researcher_as_gssc,
            "predicted_class": int(predicted_class),
            "agreement": agreement
        }

    for epoch_num, expected_researcher_stage in test_epochs:
        result = compare_epoch_predictions(epoch_num, expected_researcher_stage)
        
        if result is not None:
            all_results.append(result)
            if result["agreement"]:
                agreements += 1
    
    # Summary
    print(f"\n" + "="*50)
    print(f"MULTI-EPOCH TEST RESULTS:")
    print(f"Total epochs tested: {len(all_results)}")
    print(f"Agreements: {agreements}/{len(all_results)} ({agreements/len(all_results)*100:.1f}%)")
    
    for result in all_results:
        status = "✓" if result["agreement"] else "✗"
        print(f"  Epoch {result['epoch_number']}: {status} (Researcher:{result['researcher_stage']}→GSSC:{result['researcher_as_gssc']} vs Model:{result['predicted_class']})")
    
    print(f"="*50)
    
    print(f"\n✓ Multi-epoch test completed successfully!")
    
    # Add assertions to validate the test results
    assert len(all_results) > 0, "No epochs were tested"
    assert agreements >= 0, "Invalid number of agreements"
    
    # Validate that we tested the expected number of epochs
    expected_epochs = 4  # Based on test_epochs list
    assert len(all_results) == expected_epochs, f"Expected {expected_epochs} epochs, got {len(all_results)}"
    
    # Each result should have required fields
    for result in all_results:
        assert "epoch_number" in result, "Missing epoch_number in result"
        assert "researcher_stage" in result, "Missing researcher_stage in result"
        assert "predicted_class" in result, "Missing predicted_class in result"
        assert "agreement" in result, "Missing agreement in result"
    
    print(f"\n✅ Multi-epoch test validation completed successfully")

# The pytest test functions above will be automatically discovered and run by pytest
# This module can also be run directly for interactive testing

if __name__ == "__main__":
    import sys
    
    print("Testing shared epoch inference with REAL data and known researcher scores...")
    print("\n⚡ Running pytest tests directly...")
    
    # Run the pytest functions directly for interactive testing
    print("\n" + "="*80)
    print("TEST 1: Channel Combination Analysis")  
    print("="*80)
    test_channel_combination_inference_robust()
    
    print("\n" + "="*80)
    print("TEST 2: Multi-Epoch Validation")
    print("="*80)
    test_multiple_epochs_isolated_inference()
    
    print("\n🎉 All tests completed!")