import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import h5py
import pytest
from gssc_local.montage import Montage
from gssc_local.realtime_with_restart.processor_improved import SignalProcessor
from gssc_local.convert_csv_to_fif import convert_csv_to_raw

def test_predict_sleep_stage():
    """Test the predict_sleep_stage functionality of SignalProcessor using real data"""
    # Setup paths - using paths relative to workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_file_path = os.path.join(workspace_root, 'gssc_local', 'tests', 'test_data', 'BrainFlow-RAW_test.csv')
    mat_file_path = os.path.join(workspace_root, 'gssc_local', 'tests', 'test_data', 'scoring.mat')
    
    # Test parameters
    start = 30  # Start at 30 seconds
    end = 59.99  # End at 60 seconds
    montage = Montage.default_sleep_montage()
    eeg_channels = ['C4', 'O1', 'O2']
    eog_channels = []
    
    # Load and prepare the data
    raw = convert_csv_to_raw(csv_file_path)
    raw_sliced = raw.copy().crop(tmin=start, tmax=end)
    
    # Get channel indices
    channel_names = raw_sliced.ch_names
    eeg_indices = [i for i, ch in enumerate(channel_names) if ch in eeg_channels]
    eog_indices = [i for i, ch in enumerate(channel_names) if ch in eog_channels]
    
    # Load expected stages from .mat file
    with h5py.File(mat_file_path, 'r') as file:
        stage_data_group = file['stageData']
        sleep_stages = stage_data_group['stages'][:].flatten()
        stage_time_data_minutes = stage_data_group['stageTime'][:].flatten()
        stage_time_data_seconds = stage_time_data_minutes * 60
    
    # Get expected stages for the time window
    epoch_duration = 30
    start_epoch = int(start // epoch_duration)
    expected_stages = sleep_stages[start_epoch:2]  # Using same logic as numpy_single_epoch.py
    
    # Initialize processor and get data
    processor = SignalProcessor()
    numpy_data = raw_sliced.get_data()
    
    # Get combinations and hidden states
    eeg_eog_combo_dict = processor.get_index_combinations(eeg_indices, eog_indices)
    hiddens = processor.make_hiddens(len(eeg_eog_combo_dict))
    
    # Get predictions
    predicted_class, class_probs, new_hiddens = processor.predict_sleep_stage(
        numpy_data, 
        eeg_eog_combo_dict, 
        hiddens
    )
    
    # Basic assertions
    assert isinstance(predicted_class, int), "predicted_class should be an integer"
    assert isinstance(class_probs, np.ndarray), "class_probs should be a numpy array"
    assert len(new_hiddens) == len(eeg_eog_combo_dict), "Number of hidden states should match number of combinations"
    
    # Check predicted class is valid sleep stage (0-4)
    assert 0 <= predicted_class <= 4, "Predicted class should be between 0 and 4"
    
    # Check class probabilities
    assert class_probs.shape[0] == len(eeg_eog_combo_dict), "Number of probability rows should match number of combinations"
    assert class_probs.shape[2] == 5, "Class probabilities should have 5 classes"
    assert np.allclose(np.sum(class_probs, axis=2), 1.0), "Class probabilities should sum to 1"
    
    # Compare with expected stages
    accuracy = processor.compare_sleep_stages(np.array([predicted_class]), expected_stages, verbose=False)
    assert accuracy > 0, "Accuracy should be greater than 0"

if __name__ == '__main__':
    # Run the test directly when file is executed
    test_predict_sleep_stage()
    print("Test completed successfully!")
