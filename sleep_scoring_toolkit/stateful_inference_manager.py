"""
Stateful inference management for sleep stage classification.

This module provides StatefulInferenceManager for managing persistent hidden states
across epochs in both batch processing and real-time inference scenarios.
"""

from sleep_scoring_toolkit.core.epoch_inference import (
    infer_sleep_stage_for_epoch
)


class StatefulInferenceManager:
    """Manages stateful sleep stage inference with persistent hidden states across epochs."""
    
    def __init__(self, signal_processor, montage, eeg_channels_for_scoring, eog_channels_for_scoring, num_buffers=1):
        # Convert None to empty list for convenience while requiring at least one modality
        if eeg_channels_for_scoring is None:
            eeg_channels_for_scoring = []
        if eog_channels_for_scoring is None:
            eog_channels_for_scoring = []
            
        if not eeg_channels_for_scoring and not eog_channels_for_scoring:
            raise ValueError("At least one of eeg_channels_for_scoring or eog_channels_for_scoring must be non-empty")
            
        self.signal_processor = signal_processor
        self.montage = montage
        self.eeg_channels_for_scoring = eeg_channels_for_scoring
        self.eog_channels_for_scoring = eog_channels_for_scoring
        self.num_buffers = num_buffers
        
        # Convert channel names to positions and validate against montage
        self.eeg_positions, self.eog_positions = montage.validate_and_convert_channel_names(
            eeg_channels_for_scoring, eog_channels_for_scoring
        )
        self.index_combinations = signal_processor.get_index_combinations(self.eeg_positions, self.eog_positions)
        num_combinations = len(self.index_combinations)
        
        # Create hidden states for each buffer
        if num_buffers == 1:
            # Single buffer - maintain backward compatibility
            self.buffer_hidden_states = {0: signal_processor.make_buffer_hidden_states(num_combinations)}
        else:
            # Multiple buffers - use existing multi-buffer function
            multi_buffer_states = signal_processor.make_multi_buffer_hidden_states(num_buffers, num_combinations)
            self.buffer_hidden_states = {i: multi_buffer_states[i] for i in range(num_buffers)}
        
    def process_epoch(self, epoch_data_keyed, buffer_id=0):
        """Process an epoch using persistent hidden states (realistic predictions)."""
        if buffer_id not in self.buffer_hidden_states:
            raise ValueError(f"Invalid buffer_id {buffer_id}. Valid range: 0-{self.num_buffers-1}")
            
        # Use buffer-specific hidden states for temporal context
        hidden_states = self.buffer_hidden_states[buffer_id]
        predicted_class, class_probs, new_hidden_states = infer_sleep_stage_for_epoch(
            epoch_data_keyed, 
            self.montage, 
            self.signal_processor, 
            hidden_states,  # ← Buffer-specific hidden states with temporal context
            self.eeg_positions,  # ← Use pre-computed channel positions
            self.eog_positions   # ← Eliminates redundant computation
        )
        
        # Update buffer-specific hidden states for next epoch
        self.buffer_hidden_states[buffer_id] = new_hidden_states
        
        return predicted_class, class_probs, new_hidden_states
    
    def reset_buffer_hidden_states(self, buffer_id=None):
        """Reset hidden states for a specific buffer or all buffers (e.g., after a gap in data)."""
        # Use pre-computed channel combinations (eliminates redundant computation)
        num_combinations = len(self.index_combinations)
        
        if buffer_id is None:
            # Reset all buffers
            for bid in self.buffer_hidden_states.keys():
                self.buffer_hidden_states[bid] = self.signal_processor.make_buffer_hidden_states(num_combinations)
        else:
            # Reset specific buffer
            if buffer_id not in self.buffer_hidden_states:
                raise ValueError(f"Invalid buffer_id {buffer_id}. Valid range: 0-{self.num_buffers-1}")
            self.buffer_hidden_states[buffer_id] = self.signal_processor.make_buffer_hidden_states(num_combinations)
            
    def reset_hidden_states(self):
        """Legacy method - reset all hidden states (backward compatibility)."""
        self.reset_buffer_hidden_states()