"""
Shared core logic for sleep stage inference using GSSC model.

This module extracts the core epoch processing logic from DataManager._process_epoch() 
to enable reuse between real-time and batch processing systems.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import NumPyDataWithBrainFlowDataKey


def infer_sleep_stage_for_epoch(
    epoch_data_keyed: NumPyDataWithBrainFlowDataKey, 
    montage,
    signal_processor,
    hidden_states: List[torch.Tensor],
    eeg_combo_pk: Optional[List[int]] = None,
    eog_combo_pk: Optional[List[int]] = None
) -> Tuple[int, np.ndarray, List[torch.Tensor]]:
    """
    Shared core logic for sleep stage inference - extracted from data_manager._process_epoch() lines 592-631
    
    Args:
        epoch_data_keyed: Epoch data with channel mapping structure
        montage: Montage configuration defining channel types and validation
        signal_processor: SignalProcessor instance for GSSC model inference
        hidden_states: Current hidden states for the GSSC model
        eeg_combo_pk: List of EEG board positions to use (must be explicitly provided)
        eog_combo_pk: List of EOG board positions to use (must be explicitly provided)
        
    Returns:
        Tuple of (predicted_class, class_probabilities, new_hidden_states)
    """
    
    # Require explicit channel combinations - no fallback logic to prevent hidden bugs
    if eeg_combo_pk is None or eog_combo_pk is None:
        raise ValueError(
            "eeg_combo_pk and eog_combo_pk must be explicitly provided."
        )
    
    # Validate channel indices (line 618)
    montage.validate_channel_indices_combination_types(eeg_combo_pk, eog_combo_pk)
    
    # Get index combinations and run GSSC inference (lines 624-631)
    index_combinations = signal_processor.get_index_combinations(eeg_combo_pk, eog_combo_pk)
    
    # IMPORTANT: The GSSC was mostly trained on EEG channels C3, C4, F3, F4, and using other channels
    # is unlikely to improve accuracy and could even make accuracy worse. For EOG, the GSSC was trained
    # on the HEOG channel (left EOG - right EOG differential), and seems to also perform well with left
    # and/or right EOG alone (without the subtraction). Effectiveness of using VEOG is unknown and not recommended.
    predicted_class, class_probs, new_hidden_states = signal_processor.predict_sleep_stage(
        epoch_data_keyed,
        index_combinations,
        hidden_states
    )
    
    return predicted_class, class_probs, new_hidden_states


