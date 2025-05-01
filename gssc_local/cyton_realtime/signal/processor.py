import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from gssc.infer import ArrayInfer
from gssc.utils import loudest_vote

class SignalProcessor:
    """Handles signal processing and sleep stage prediction"""
    def __init__(self, use_cuda=False, gpu_idx=None):
        self.infer = ArrayInfer(
            net=None,  # Use default network
            con_net=None,  # Use default context network
            use_cuda=use_cuda,
            gpu_idx=gpu_idx
        )
        
    def resample_tensor(self, data, target_length):
        """Resample a tensor to a target length using interpolation"""
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        resampled = F.interpolate(
            data, 
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        if len(data.shape) == 2:
            resampled = resampled.squeeze(0)
            
        return resampled
    
    def make_combo_dictionary(self, epoch_data, eeg_index, eog_index):
        """Create input dictionary for EEG and EOG data"""
        input_dict = {}
        epoch_tensor = (torch.tensor(epoch_data, dtype=torch.float32) 
                       if not isinstance(epoch_data, torch.Tensor) 
                       else epoch_data.float())
        
        if eeg_index is not None:
            eeg_data = epoch_tensor[eeg_index].unsqueeze(0).unsqueeze(0)
            input_dict['eeg'] = eeg_data
        
        if eog_index is not None:
            eog_data = epoch_tensor[eog_index].unsqueeze(0).unsqueeze(0)
            input_dict['eog'] = eog_data
        
        return input_dict
    
    def get_index_combinations(self, eeg_indices, eog_indices):
        """Generate all possible combinations of EEG and EOG indices"""
        combinations = []
        
        # Add EEG-EOG pairs
        for eeg_idx in eeg_indices:
            for eog_idx in eog_indices:
                combinations.append([eeg_idx, eog_idx])
        
        # Add EEG only combinations
        for eeg_idx in eeg_indices:
            combinations.append([eeg_idx, None])
        
        # Add EOG only combinations
        for eog_idx in eog_indices:
            combinations.append([None, eog_idx])
            
        return combinations
    
    def prepare_input_data(self, epoch_data):
        """Prepare input data for prediction"""
        # Hardcoded indices for now - could be made configurable
        index_combinations = self.get_index_combinations([0, 1, 2], [3])
        
        # Create input dictionaries
        input_dict_list = []
        for eeg_idx, eog_idx in index_combinations:
            input_dict = self.make_combo_dictionary(epoch_data, eeg_idx, eog_idx)
            input_dict_list.append(input_dict)
        
        # Resample all inputs to 2560
        resampled_dict_list = []
        for input_dict in input_dict_list:
            new_dict = {}
            if 'eeg' in input_dict:
                new_dict['eeg'] = self.resample_tensor(input_dict['eeg'], 2560)
            if 'eog' in input_dict:
                new_dict['eog'] = self.resample_tensor(input_dict['eog'], 2560)
            resampled_dict_list.append(new_dict)
            
        return resampled_dict_list
    
    def predict_sleep_stage(self, epoch_data, hidden_states):
        """Predict sleep stage from epoch data"""
        # Prepare input data
        input_dict_list = self.prepare_input_data(epoch_data)
        
        # Get predictions for each combination
        results = []
        for i, input_dict in enumerate(input_dict_list):
            logits, res_logits, hidden_state = self.infer.infer(input_dict, hidden_states[i])
            results.append([logits, res_logits, hidden_state])
        
        # Combine logits
        all_combo_logits = np.stack([
            results[i][0].numpy() for i in range(len(results))
        ])
        
        # Get final prediction
        final_predicted_class = loudest_vote(all_combo_logits)
        new_hidden_states = [result[2] for result in results]
        
        return final_predicted_class, new_hidden_states 