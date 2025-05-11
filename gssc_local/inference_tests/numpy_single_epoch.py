"""
This script is used to test  one single epoch of data with processor_improved.py. This is essentially
a test to see if the processor_improved.py is working as expected. Processor_improved.py is combination
of gsssc_helper.py and processor.py. We should be able to delete both of these files and just use
processor_improved.py for all inference needs.
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# import montage class
import mne
from gssc.infer import EEGInfer
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# import realtime inference from gssc_array_infer.py in this directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from montage import Montage
from gssc_helper import realtime_inference, make_hiddens, preprocess_eeg_epoch_for_gssc, prepare_input, get_predicted_classes, get_predicted_classes_and_probabilities, get_results_for_each_combo, make_eeg_eog_combinations, make_infer, compare_sleep_stages
from convert_csv_to_fif import convert_csv_to_raw, save_raw_to_fif, convert_csv_to_fif
from gssc_local.realtime_with_restart.processor_improved import SignalProcessor

csv_file_path = 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.csv'
start = 30
end = 59.99
montage = Montage.default_sleep_montage()
montage_ch_names = montage.get_channel_labels()
eeg_channels = ['C4', 'O1', 'O2']  #  Adjust based on your data
eog_channels = []  # Adjust based on your data

# Then read it to get the correct channel names
raw = convert_csv_to_raw(csv_file_path)
raw_sliced = raw.copy().crop(tmin=start, tmax=end) 
# Path to save the .fif file

channel_names = raw_sliced.ch_names

# check the number of samples in the raw file
print(raw_sliced.n_times)

# Create indices from the newly saved file
eeg_indices = [i for i, ch in enumerate(channel_names) if ch in eeg_channels]
eog_indices = [i for i, ch in enumerate(channel_names) if ch in eog_channels]

# Initialize the EEGInfer class
eeg_infer = EEGInfer()



# Path to your .mat file
mat_file_path = 'data/realtime_inference_test/scoring.mat'

# Open the .mat file
with h5py.File(mat_file_path, 'r') as file:
    # Access the 'stageData' group
    if 'stageData' in file:
        stage_data_group = file['stageData']
        
        # Read the 'stages' dataset
        if 'stages' in stage_data_group:
            sleep_stages = stage_data_group['stages'][:].flatten()  # Flatten to 1D array
        else:
            raise ValueError("Dataset 'stages' not found in 'stageData'.")
        
        # Read the 'stageTime' dataset to determine the time intervals
        if 'stageTime' in stage_data_group:
            stage_time_data_minutes = stage_data_group['stageTime'][:].flatten()  # Flatten to 1D array
            stage_time_data_seconds = stage_time_data_minutes * 60  # Convert to seconds
        else:
            raise ValueError("Dataset 'stageTime' not found in 'stageData'.")





# Determine the stages for an interval
# Assuming each stage corresponds to a 30-second epoch
# Each stage in sleep_stages represents one 30-second epoch
# If start and end are null, return all stages except the last one

# Convert time window to epoch indices
epoch_duration = 30  # Each epoch is 30 seconds
start_epoch = int(start // epoch_duration)
end_epoch = int(end // epoch_duration)

# Select stages for the specified epochs
if start is None and end is None:
    expected_stages = sleep_stages[:-1]
else:
    # TODO: fix this. hard coded end epoch because the last epoch is not complete
    expected_stages = sleep_stages[start_epoch:2]

print("array_inference---------------")
# compare_sleep_stages([loudest_vote], expected_stages, verbose=False)

# using processor_improved.py
# get 30 seconds of data from a csv file in numpy array
# get number of samples in the raw data
print(f"Number of samples in raw data: {raw_sliced.n_times}")
numpy_data = raw_sliced.get_data()

# create a processor object
processor = SignalProcessor()


# get the combination dictionary
eeg_eog_combo_dict = processor.get_index_combinations(eeg_indices, eog_indices)

hiddens = processor.make_hiddens(len(eeg_eog_combo_dict))

# get the predicted classes
predicted_classes, class_probs, new_hiddens = processor.predict_sleep_stage(numpy_data, eeg_eog_combo_dict, hiddens)

# print the predicted classes
processor.compare_sleep_stages(np.array([predicted_classes]), expected_stages, verbose=False)

# print the class probabilities
processor.print_class_probabilities(np.array([predicted_classes]), class_probs)













