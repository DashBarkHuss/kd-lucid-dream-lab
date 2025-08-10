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
from gssc_local.convert_csv_to_fif import convert_csv_to_raw, save_raw_to_fif, convert_csv_to_fif, convert_numpy_to_raw

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
        Create hidden states for a new prediction sequence.
        
        Args:
            num_combinations (int): Number of channel combinations
            
        Returns:
            torch.Tensor: Initialized hidden states tensor
        """
        return torch.zeros(num_combinations, 10, 1, 256)
    
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
    
    def resample_to_target_frequency(self, raw: mne.io.Raw, target_sfreq: float) -> mne.io.Raw:
        """
        Resample raw data to target frequency with proper upsampling to 1000 Hz.
        Uses MNE's resampling functionality for high-quality resampling.
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            target_sfreq (float): Target sampling frequency
            
        Returns:
            mne.io.Raw: Resampled raw data
        """
        current_sfreq = raw.info['sfreq']
        
        # First upsample to 1000 Hz if needed
        if current_sfreq < 1000:
            raw = raw.copy().resample(1000)
            current_sfreq = 1000
            
        # Then downsample to target frequency if needed
        if current_sfreq != target_sfreq:
            raw = raw.copy().resample(target_sfreq)
            
        return raw
    
    def bandpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply bandpass filter to raw data using MNE's filtering.
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            
        Returns:
            mne.io.Raw: Filtered raw data
        """
        # Store a copy of the data before filtering
        data_before = raw.get_data().copy()
        
        filter_band = [None, None]
        if round(raw.info["highpass"], 2) < 0.3:
            filter_band[0] = 0.3
        if round(raw.info["lowpass"], 2) > 30.:
            filter_band[1] = 30.
        raw = raw.copy().filter(*filter_band)
        
        # Get data after filtering
        data_after = raw.get_data()
        
        # Check if data actually changed
        if np.array_equal(data_before, data_after):
            warnings.warn("WARNING: Filtering did not change the data. This might indicate an issue with the filtering process.")
        else:
            print(f"Filtering changed the data. Max difference: {np.max(np.abs(data_before - data_after))}")
            
        if round(raw.info["highpass"], 2) != 0.3:
            warnings.warn("WARNING: GSSC was trained on data with a highpass "
                        "filter of 0.3Hz. These data have a highpass filter "
                        f"of {raw.info['highpass']}Hz")
        if round(raw.info["lowpass"], 2) != 30.:
            warnings.warn("WARNING: GSSC was trained on data with a lowpass "
                        "filter of 30Hz. These data have a lowpass filter "
                        f"of {raw.info['lowpass']}Hz")
        return raw
    
    def preprocess_epoch(self, epoch_data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a 30-second epoch of EEG data for sleep stage classification.
        
        Args:
            epoch_data (torch.Tensor): Raw EEG data
            
        Returns:
            numpy.ndarray: Preprocessed raw data
            
        Raises:
            ValueError: If data length is invalid
        """
        raw = convert_numpy_to_raw(epoch_data)
        expected_samples = int(30 * raw.info['sfreq'])
        # Validate epoch length
        if epoch_data.shape[1] != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples for 30s epoch, got {epoch_data.shape[1]}")
        # the target sfreq is 2560/30
        target_sfreq = 2560/30
        # Process the data
        raw = self.resample_to_target_frequency(raw, target_sfreq)
        raw = self.bandpass_filter(raw)
        # Get data and apply z-scoring
        data = raw.get_data()
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
        from gssc_local.realtime_with_restart.channel_mapping import NumPyDataWithBrainFlowDataKey
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
    
    def predict_sleep_stage(self, epoch_data_keyed, index_combinations: List[Tuple[Optional[int], Optional[int]]], hidden_states: List[torch.Tensor] = None) -> Tuple[int, np.ndarray, List[torch.Tensor]]:
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