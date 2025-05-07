"""
The script compares sleep stage classification results of the csv from different inference methods.
It processes EEG data from the fif file, and runs both MNE-based and array-based GSSC inference
algorithms. The script then compares the predicted sleep stages with ground truth data from a .mat
file, calculating accuracy metrics and providing detailed analysis of classification performance
across different sleep stages.

This file uses gssc_array_inference_with_fif.py to get the predicted classes.
"""

# import montage class
from montage import Montage
import mne
from gssc.infer import EEGInfer
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# import realtime inference from gssc_array_infer.py in this directory
from gssc_helper import realtime_inference, compare_sleep_stages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from convert_csv_to_fif import convert_csv_to_raw, save_raw_to_fif, convert_csv_to_fif



csv_file_path = '/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.csv'
# Read the .set file
raw_csv = convert_csv_to_raw(csv_file_path)

# resample the data to 1000 Hz. 125hz caused a floating point error in prepare_inst
raw_csv.resample(1000)


start = 3000
end = 6000

# Slice the data 
raw_csv_sliced = raw_csv.copy().crop(tmin=start, tmax=end) 
# get channel labels from default montage
montage = Montage.default_sleep_montage()
montage_ch_names = montage.get_channel_labels()
eeg_channels = ['C4', 'O1', 'O2']  #  Adjust based on your data
eog_channels = []  # Adjust based on your data


channel_names = raw_csv_sliced.ch_names

# Create indices from the newly saved file
eeg_indices = [i for i, ch in enumerate(channel_names) if ch in eeg_channels]
eog_indices = [i for i, ch in enumerate(channel_names) if ch in eog_channels]

# Initialize the EEGInfer class
eeg_infer = EEGInfer()

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
    expected_stages = sleep_stages[start_epoch:end_epoch]

# compare the inferred sleep stages with the expected sleep stages





# all_predicted_classes = realtime_inference(fif_file_path)
loudest_votes, predicted_classes, class_probs = realtime_inference(raw_csv_sliced, eeg_channels, eog_channels)

for i in range(len(predicted_classes)):
        print(f"  Predicted class: {int(predicted_classes[i][0])}")
        print("  Class probabilities:")
        for class_idx, prob in enumerate(class_probs[i][0]):
            print(f"    Class {class_idx}: {float(prob[0])*100:.2f}%")

# print("mne-eeglab---------------")
# compare_sleep_stages(inferred_stages, expected_stages, verbose=False)
# print("old---------------")
# compare_sleep_stages(all_predicted_classes, expected_stages, verbose=False)
print("array_inference---------------")
compare_sleep_stages(loudest_votes, expected_stages, verbose=False)

