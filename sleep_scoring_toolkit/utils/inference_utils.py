"""
Utility functions for sleep stage inference validation and researcher score conversion.

This module provides helper functions for validating inference results,
converting between different sleep stage scoring systems, and formatting results.
"""

import numpy as np
from sleep_scoring_toolkit.constants import ResearcherStages, GSSCStages


def validate_inference_results(predicted_class, class_probs, new_hidden_states):
    """Validate the structure of inference results."""
    assert isinstance(predicted_class, (int, np.integer)), f"Expected int, got {type(predicted_class)}"
    assert GSSCStages.is_valid(predicted_class), f"Sleep stage {predicted_class} out of range"
    assert isinstance(class_probs, np.ndarray), f"Expected ndarray, got {type(class_probs)}"
    assert class_probs.shape[-1] == GSSCStages.NUM_STAGES, f"Expected {GSSCStages.NUM_STAGES} sleep stage probabilities"
    assert len(new_hidden_states) > 0, f"Expected at least 1 hidden state, got {len(new_hidden_states)}"


def convert_researcher_stage_to_gssc(researcher_score):
    """Convert researcher stage to GSSC scale for comparison."""
    if researcher_score == ResearcherStages.REM:
        return GSSCStages.REM  # Researcher 5 → GSSC 4
    elif researcher_score == ResearcherStages.WAKE:
        return GSSCStages.WAKE  # Researcher 0 → GSSC 0
    elif researcher_score == ResearcherStages.N1:
        return GSSCStages.N1  # Researcher 1 → GSSC 1
    elif researcher_score == ResearcherStages.N2:
        return GSSCStages.N2  # Researcher 2 → GSSC 2
    elif researcher_score == ResearcherStages.N3:
        return GSSCStages.N3  # Researcher 3 → GSSC 3
    else:
        return None  # Unknown/unsupported researcher stage


def validate_researcher_score(researcher_score, expected_stage, epoch_num, timestamp):
    """Validate researcher score matches expectations and timing is correct."""
    if researcher_score is None:
        raise ValueError(f"No researcher score found for epoch {epoch_num} at timestamp {timestamp}. "
                        f"This suggests timing misalignment exceeds tolerance (±2 seconds)")
        
    if researcher_score != expected_stage:
        raise ValueError(f"Expected researcher stage {expected_stage}, got {researcher_score} "
                        f"for epoch {epoch_num}")
        
    return True


def print_epoch_results(epoch_num, stage_name, expected_stage, predicted_class, researcher_score, researcher_as_gssc, agreement):
    """Print formatted results for a single epoch test."""
    print(f"\n" + "-"*50)
    print(f"Testing Epoch {epoch_num} (Expected: {stage_name}/Researcher Stage {expected_stage})")
    print(f"-"*50)
    
    print(f"Researcher score: {researcher_score} ({ResearcherStages.to_name(researcher_score)})")
    print(f"Model prediction: {predicted_class} ({GSSCStages.to_name(predicted_class)})")
    
    if researcher_as_gssc is not None:
        print(f"Agreement: {'✓' if agreement else '✗'} (researcher {researcher_score}→{researcher_as_gssc} vs model {predicted_class})")
    else:
        print(f"Cannot compare: researcher stage {researcher_score} not supported")