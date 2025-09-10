"""
Improved signal processor for sleep stage classification using GSSC model.
Combines functionality from processor.py and gssc_helper.py with enhanced features.
"""
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import mne
import warnings
from gssc.infer import ArrayInfer
from gssc.utils import loudest_vote, epo_arr_zscore, permute_sigs, prepare_inst
import scipy.signal
from scipy.signal import butter, lfilter
# import convert_csv_to_fif.py
from sleep_scoring_toolkit.convert_csv_to_fif import convert_csv_to_raw, save_raw_to_fif, convert_csv_to_fif, convert_numpy_to_raw

class SignalProcessor:
    """Handles signal processing and sleep stage prediction with enhanced features"""
    
    def __init__(self, use_cuda=False, gpu_idx=None):
        """
        Initialize the signal processor.
        
        Args:
            use_cuda (bool): Whether to use CUDA for GPU acceleration
            gpu_idx (Optional[int]): Specific GPU index to use
        """
        self.infer = ArrayInfer(
            net=None,  # Use default network
            con_net=None,  # Use default context network
            use_cuda=False,  # Explicitly disable CUDA
            gpu_idx=None  # No GPU index needed
        )
        
    def make_hiddens(self, num_combinations: int) -> torch.Tensor:
        """
        Create hidden states for a new prediction sequence (single tensor format).
        
        Args:
            num_combinations (int): Number of channel combinations
            
        Returns:
            torch.Tensor: Initialized hidden states tensor (num_combinations, 10, 1, 256)
        """
        return torch.zeros(num_combinations, 10, 1, 256)
    
    def make_buffer_hidden_states(self, num_combinations: int = 7) -> List[torch.Tensor]:
        """
        Create hidden states for buffer-based processing (list format).
        Used in real-time processing with multiple buffers.
        
        Args:
            num_combinations (int): Number of channel combinations (default: 7)
            
        Returns:
            List[torch.Tensor]: List of initialized hidden state tensors
        """
        return [torch.zeros(10, 1, 256) for _ in range(num_combinations)]
    
    def make_multi_buffer_hidden_states(self, num_buffers: int = 6, num_combinations: int = 7) -> List[List[torch.Tensor]]:
        """
        Create hidden states for multi-buffer processing (nested list format).
        Used in real-time processing with multiple buffers at different time offsets.
        
        Args:
            num_buffers (int): Number of time-offset buffers (default: 6)
            num_combinations (int): Number of channel combinations per buffer (default: 7)
            
        Returns:
            List[List[torch.Tensor]]: Nested list of hidden states [buffer_id][combination_id]
        """
        return [self.make_buffer_hidden_states(num_combinations) for _ in range(num_buffers)]
    
    def make_combo_dictionary(self, epoch_data_keyed, eeg_key: Optional[int], eog_key: Optional[int]) -> Dict[str, torch.Tensor]:
        """
        Create input dictionary for EEG and EOG data using BrainFlow keys.
        
        Args:
            epoch_data_keyed (DataWithBrainFlowDataKey): Wrapper containing processed data with channel mapping.
                The wrapper.data should be torch.Tensor or numpy.ndarray with processed EEG/EOG data.
            eeg_key (Optional[int]): EEG channel BrainFlow key (board position)
            eog_key (Optional[int]): EOG channel BrainFlow key (board position)
            
        Returns:
            Dict[str, torch.Tensor]: Input dictionary with processed data
        """
        input_dict = {}
        
        if eeg_key is not None:
            # Get EEG data by BrainFlow key
            eeg_data = epoch_data_keyed.get_by_key(eeg_key)
            # Convert to tensor with proper validation and reshape for model input
            eeg_tensor = (torch.tensor(eeg_data, dtype=torch.float32) 
                         if not isinstance(eeg_data, torch.Tensor) 
                         else eeg_data.float()).reshape(1, 1, -1)
            input_dict['eeg'] = eeg_tensor
        
        if eog_key is not None:
            # Get EOG data by BrainFlow key
            eog_data = epoch_data_keyed.get_by_key(eog_key)
            # Convert to tensor with proper validation and reshape for model input
            eog_tensor = (torch.tensor(eog_data, dtype=torch.float32) 
                         if not isinstance(eog_data, torch.Tensor) 
                         else eog_data.float()).reshape(1, 1, -1)
            input_dict['eog'] = eog_tensor
        
        return input_dict
    
    def get_index_combinations(self, eeg_indices: List[int], eog_indices: List[int]) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Generate all valid combinations of EEG and EOG indices.
        
        Args:
            eeg_indices (List[int]): List of EEG channel indices
            eog_indices (List[int]): List of EOG channel indices
            
        Returns:
            List[Tuple[Optional[int], Optional[int]]]: List of valid channel combinations
        """
        combinations = []
        # Include None as an option for both EEG and EOG
        for eeg_index in eeg_indices + [None]:
            for eog_index in eog_indices + [None]:
                # Skip if both are None
                if eeg_index is None and eog_index is None:
                    continue
                combinations.append((eeg_index, eog_index))
        return combinations
    
    def validate_epoch_length(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Validate that the data has exactly 30 seconds of samples.
        
        Args:
            data (np.ndarray): Input data array
            sfreq (float): Sampling frequency
            
        Returns:
            np.ndarray: Validated data array
            
        Raises:
            ValueError: If data length is invalid
        """
        expected_samples = int(30 * sfreq)
        if data.shape[1] != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples for 30s epoch, got {data.shape[1]}")
        return data
    
    def resample_to_exact_samples(self, data: np.ndarray, target_samples: int) -> np.ndarray:
        """
        Resample single epoch data to have exactly the target number of samples using scipy resampling.
        
        This is the primary resampling method that converts EEG data from any sampling rate
        directly to exactly 2560 samples (30 seconds at 85.33 Hz) as required by the GSSC model.
        
        Args:
            data (np.ndarray): Input data with shape (n_channels, n_samples)
            target_samples (int): Exact number of samples required (2560 for GSSC)
            
        Returns:
            np.ndarray: Resampled data with shape (n_channels, target_samples)
        """
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_samples), got shape: {data.shape}")
        
        current_samples = data.shape[1]
        if current_samples == target_samples:
            return data  # Already correct size
        
        n_channels = data.shape[0]
        resampled_data = np.zeros((n_channels, target_samples))
        for ch in range(n_channels):
            resampled_data[ch, :] = scipy.signal.resample(data[ch, :], target_samples)
        return resampled_data

    def bandpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply GSSC-required bandpass filter (0.3-30 Hz) to raw EEG data.
        
        The GSSC model was trained on data bandpass filtered at 0.3-30 Hz.
        This method expects unfiltered raw data and applies the exact filters
        required by GSSC.
        
        Args:
            raw (mne.io.Raw): Unfiltered raw EEG data
            
        Returns:
            mne.io.Raw: Filtered raw data matching GSSC training specifications
            
        Raises:
            ValueError: If input data is already filtered
        """
        # Validate that input data is unfiltered
        if raw.info["highpass"] != 0.0:
            raise ValueError(f"Expected unfiltered data (highpass=0.0), got highpass={raw.info['highpass']} Hz")
        
        # Accept lowpass at Nyquist frequency as "unfiltered" - this is just the theoretical sampling limit
        nyquist_freq = raw.info['sfreq'] / 2
        if (raw.info["lowpass"] is not None and 
            raw.info["lowpass"] != nyquist_freq and
            raw.info["lowpass"] < 100.0):
            raise ValueError(f"Input data appears pre-filtered with lowpass={raw.info['lowpass']} Hz. Expected unfiltered raw data (Nyquist={nyquist_freq} Hz).")
        
        # Store a copy of the data before filtering
        data_before = raw.get_data().copy()
        
        # Apply GSSC-required bandpass filter
        raw = raw.copy().filter(l_freq=0.3, h_freq=30.0)
        
        # Get data after filtering
        data_after = raw.get_data()
        
        # Check if data actually changed
        if np.array_equal(data_before, data_after):
            warnings.warn("WARNING: Filtering did not change the data. This might indicate an issue with the filtering process.")
            
        # Validate that MNE applied the filters correctly
        if raw.info["highpass"] != 0.3:
            raise ValueError(f"Filter application failed: expected highpass=0.3 Hz, got {raw.info['highpass']} Hz")
        if raw.info["lowpass"] != 30.0:
            raise ValueError(f"Filter application failed: expected lowpass=30.0 Hz, got {raw.info['lowpass']} Hz")
            
        return raw
    
    def preprocess_epoch(self, epoch_data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a 30-second epoch of EEG data for sleep stage classification.
        
        Args:
            epoch_data (torch.Tensor): Raw EEG data
            
        Returns:
            numpy.ndarray: Preprocessed raw data with exactly 2560 samples
            
        Raises:
            ValueError: If data length is invalid
        """
        raw = convert_numpy_to_raw(epoch_data)
        expected_samples_for_freq = int(30 * raw.info['sfreq'])
        
        # Validation: ensure we have at least target samples and catch obvious anomalies
        sample_count = epoch_data.shape[1]
        min_samples = 2560  # Must have at least target samples for resampling
        max_samples = int(expected_samples_for_freq * 10)  # Generous upper bound to catch major errors
        
        if sample_count < min_samples:
            raise ValueError(f"Epoch sample count {sample_count} too low - need at least {min_samples} samples for resampling")
        if sample_count > max_samples:
            raise ValueError(f"Epoch sample count {sample_count} suspiciously high ({max_samples} max suggests major data issue)")
        
        # GSSC requires exactly 2560 samples per 30-second epoch (85.33 Hz effective rate)
        # Reason: GSSC training methodology downsampled all data to 85.33 Hz, reducing 30s sections to 2560 samples
        target_samples = 2560
        
        # Apply bandpass filter first (on original sampling rate)
        raw = self.bandpass_filter(raw)
        
        # Get data and apply direct resampling to exact target samples
        data = raw.get_data()
        data = self.resample_to_exact_samples(data, target_samples)
        
        # Apply z-scoring after resampling
        data = epo_arr_zscore(data)
            
        return data
    
    def prepare_input_data(self, epoch_data_keyed, index_combinations: List[Tuple[Optional[int], Optional[int]]]) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare input data for prediction with comprehensive validation.
        
        Args:
            epoch_data_keyed (DataWithBrainFlowDataKey): Wrapper containing 30-second chunk of EEG/EOG data
                with structured channel mapping using BrainFlow board positions as keys.
                The wrapper.data should be torch.Tensor or numpy.ndarray with shape [n_channels, n_samples]
            index_combinations (List[Tuple[Optional[int], Optional[int]]]): List of (eeg_board_position, eog_board_position) combinations
            
        Returns:
            List[Dict[str, torch.Tensor]]: List of processed input dictionaries
            
        Raises:
            ValueError: If input validation fails
        """
        # Input validation - check wrapper structure
        if not hasattr(epoch_data_keyed, 'data') or not hasattr(epoch_data_keyed, 'get_by_key'):
            raise ValueError("epoch_data_keyed must be DataWithBrainFlowDataKey with data and get_by_key attributes")
        
        # Input validation - check underlying data type
        if not isinstance(epoch_data_keyed.data, (torch.Tensor, np.ndarray)):
            raise ValueError("epoch_data_keyed.data must be torch.Tensor or numpy.ndarray")
            
        # Validate channel indices
        if not index_combinations:
            raise ValueError("No valid channel combinations found")
            
        # Preprocess the full data
        sfreq = 2560/30  # Target sampling rate
        processed_epoch_data = self.preprocess_epoch(epoch_data_keyed.data)
        
        # Create new wrapper with processed data but same channel mapping
        from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import NumPyDataWithBrainFlowDataKey
        processed_data_keyed = NumPyDataWithBrainFlowDataKey(
            data=processed_epoch_data,
            channel_mapping=epoch_data_keyed.channel_mapping
        )
        
        # Process each combination
        processed_dicts = []
        for eeg_key, eog_key in index_combinations:
            # Create input dictionary using make_combo_dictionary with BrainFlow keys
            input_dict = self.make_combo_dictionary(processed_data_keyed, eeg_key, eog_key)
            processed_dicts.append(input_dict)
            
        return processed_dicts
    
    def predict_sleep_stage(self, epoch_data_keyed, index_combinations: List[Tuple[Optional[int], Optional[int]]], 
                          hidden_states: List[torch.Tensor] = None) -> Tuple[int, np.ndarray, List[torch.Tensor]]:
        """
        Predict sleep stage from epoch data with enhanced error handling.
        
        Args:
            epoch_data_keyed (DataWithBrainFlowDataKey): Input data wrapper with channel mapping
            index_combinations (List[Tuple[Optional[int], Optional[int]]]): List of channel combinations
            hidden_states (List[torch.Tensor]): List of hidden states for each combination
            
        Returns:
            Tuple[int, np.ndarray, List[torch.Tensor]]: 
                - Predicted class (0-4)
                - Class probabilities (shape: [n_combinations, 5])
                - Updated hidden states
        """
     
        # Prepare input data - pass wrapper directly for key-based access
        input_dict_list = self.prepare_input_data(epoch_data_keyed, index_combinations)
        
        # Get predictions for each combination
        results = []
        for i, input_dict in enumerate(input_dict_list):
            logits, res_logits, hidden_state = self.infer.infer(input_dict, hidden_states[i])
            results.append([logits, res_logits, hidden_state])
        
        # Combine logits
        all_combo_logits = np.stack([
            results[i][0].numpy() for i in range(len(results))
        ])
        
        # Get final prediction using loudest vote
        # Extract the first (and only) element to avoid deprecation warning
        final_predicted_class = int(loudest_vote(all_combo_logits).item())
        
        # Get class probabilities
        logits_tensor = torch.tensor(all_combo_logits)
        probabilities = torch.softmax(logits_tensor, dim=-1)
        class_probabilities = probabilities.numpy()
        
        # Update hidden states
        new_hidden_states = [result[2] for result in results]
        
        return final_predicted_class, class_probabilities, new_hidden_states

    def convert_raw_full_data_to_gssc_tensor(self, raw: mne.io.Raw, epoch_duration: int = 30, filter: bool = True) -> torch.Tensor:
        """
        Convert raw MNE data to GSSC-compatible tensor format.
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            epoch_duration (int): Duration of each epoch in seconds (default: 30)
            filter (bool): Whether to apply bandpass filtering (default: True)
            
        Returns:
            torch.Tensor: Processed data tensor ready for GSSC
        """
        channel_names = raw.ch_names
        
        if filter:
            raw = self.bandpass_filter(raw)
            
        epo3, _ = prepare_inst(raw, 2560, 'back')
        data1 = epo3.get_data(picks=channel_names) * 1e+6
        
        data3 = epo_arr_zscore(data1)
        
        return torch.tensor(data3).float()
    
    def compare_sleep_stages(self, inferred_stages: np.ndarray, expected_stages: np.ndarray, verbose: bool = True) -> float:
        """
        Compare predicted sleep stages with ground truth.
        
        Args:
            inferred_stages (np.ndarray): Predicted sleep stages
            expected_stages (np.ndarray): Ground truth sleep stages
            verbose (bool): Whether to print detailed comparison
            
        Returns:
            float: Overall accuracy
            
        Raises:
            ValueError: If arrays have different lengths
        """
        sleepSMG_staging_key = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            5: 'REM',
            6: 'Movement/Undefined',
            7: 'Unscored'
        }
        gssc_staging_key = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        
        # Ensure both arrays are the same length
        if len(inferred_stages) != len(expected_stages):
            raise ValueError("Inferred and expected stage arrays must be the same length")
        
        # Remove unscored and movement/undefined epochs
        remove_epochs = np.where((expected_stages == 7) | (expected_stages == 6))[0]
        inferred_stages_cut = np.delete(inferred_stages, remove_epochs)
        expected_stages_cut = np.delete(expected_stages, remove_epochs)
        
        # Initialize counters
        matches = 0
        total = len(inferred_stages_cut)
        mismatches = []
        
        # Compare stages epoch by epoch
        print("")
        max_length = max(len(f"Epoch {len(inferred_stages)}: ") + 20, 30)
        
        for i, (inferred, expected) in enumerate(zip(inferred_stages_cut, expected_stages_cut)):
            if gssc_staging_key[inferred] == sleepSMG_staging_key[expected]:
                matches += 1
                print(f"Epoch {i+1}: Stage {gssc_staging_key[inferred]}")
            else:
                mismatches.append((i, inferred, expected))
                first_part = f"Epoch {i+1}: Stage {gssc_staging_key[inferred]}"
                print(f"{first_part:<{max_length}} NOT MATCHED --Expected Stage {sleepSMG_staging_key[expected]}")
        
        # Calculate accuracy
        accuracy = matches / total
        
        # Print summary
        print(f"\nTotal Epochs: {total}")
        print(f"Matched Epochs: {matches}")
        print(f"Mismatched Epochs: {total - matches}")
        print(f"Accuracy: {(accuracy*100):.4f}%")
        
        if verbose:
            # Calculate per-stage metrics
            stages = {
                'REM': (4, 5),
                'N1': (1, 1),
                'N2': (2, 2),
                'N3': (3, 3),
                'Wake': (0, 0)
            }
            
            for stage_name, (inferred_val, expected_val) in stages.items():
                false_pos = [m for m in mismatches if m[1] == inferred_val]
                false_neg = [m for m in mismatches if m[2] == expected_val]
                
                false_pos_prc = len(false_pos) / total
                false_neg_prc = len(false_neg) / total
                overall_accuracy = 1 - ((len(false_pos) + len(false_neg)) / total)
                
                print(f"\nFalse Positive {stage_name} Epochs: {len(false_pos)}")
                print(f"False Positive {stage_name} percentage: {(false_pos_prc * 100):.2f}%")
                print(f"False Negative {stage_name} Epochs: {len(false_neg)}")
                print(f"False Negative {stage_name} percentage: {(false_neg_prc * 100):.2f}%")
                print(f"Overall {stage_name} Accuracy: {(overall_accuracy * 100):.2f}%")
        
        print(f"\nRemoved Unscored and Movement/Undefined Epochs: {len(remove_epochs)}")
        
        return accuracy 

    def print_class_probabilities(self, predicted_classes: np.ndarray, class_probs: np.ndarray) -> None:
        """
        Print class probabilities for each predicted class in a nicely formatted table.
        
        Args:
            predicted_classes (np.ndarray): Array of predicted class indices
            class_probs (np.ndarray): Array of class probabilities for each prediction
                Shape should be [n_predictions, n_classes]
        """
        gssc_staging_key = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        
        # Print header
        print("\nClass Probabilities:")
        print("-" * 80)
        print(f"{'Epoch':<8} {'Predicted':<10} {'Wake':<10} {'N1':<10} {'N2':<10} {'N3':<10} {'REM':<10}")
        print("-" * 80)
        
        # Ensure predicted_classes is a 1D array
        predicted_classes = np.asarray(predicted_classes).flatten()
        
        # Print probabilities for each epoch
        for i, (pred_class, probs) in enumerate(zip(predicted_classes, class_probs)):
            # Convert numpy values to Python floats and format as percentages
            # Handle both single probabilities and arrays of probabilities
            if isinstance(probs, np.ndarray) and probs.ndim > 0:
                prob_strs = [f"{float(p)*100:>8.2f}%" for p in probs.flatten()]
            else:
                prob_strs = [f"{float(probs)*100:>8.2f}%"]
            print(f"{i+1:<8} {gssc_staging_key[pred_class]:<10} {' '.join(prob_strs)}")
        
        print("-" * 80) 