"""
This script convertsa csv to a fif file andraw mne so we can use it with some of the functinos
and scripts we already have. This would probably eb more modular if we just used mne raw, but
we originally coded the functions to accept a fif file and convert it to raw so whatever for now.
The script compares sleep stage classification results of the fif file from different inference methods.
It processes EEG data from the fif file, and runs both MNE-based and array-based GSSC inference
algorithms. The script then compares the predicted sleep stages with ground truth data from a .mat
file, calculating accuracy metrics and providing detailed analysis of classification performance
across different sleep stages.

This file uses gssc_array_inference_with_fif.py to get the predicted classes.
"""

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

start = 30
end = 59.999
montage = Montage.default_sleep_montage()
montage_ch_names = montage.get_channel_labels()
eeg_channels = ['C4', 'O1', 'O2']  #  Adjust based on your data
eog_channels = []  # Adjust based on your data
# eeg_channels = ['C3']  # Adjust based on your data
# eog_channels = ['L-HEOG', 'R-HEOG']  # Adjust based on your data


# Path to save the .fif file
fif_file_path = 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0_thirty_seconds.fif'

# Then read it to get the correct channel names
raw_csv_as_fif = mne.io.read_raw_fif(fif_file_path, preload=True)
channel_names = raw_csv_as_fif.ch_names

# check the number of samples in the raw file
print(raw_csv_as_fif.n_times)

# Create indices from the newly saved file
eeg_indices = [i for i, ch in enumerate(channel_names) if ch in eeg_channels]
eog_indices = [i for i, ch in enumerate(channel_names) if ch in eog_channels]

# Initialize the EEGInfer class
eeg_infer = EEGInfer()

# Specify the EEG and EOG channels
# I'm testing with only C3 to get ArrayInfer to match, usually you should use the comment out portion below
# eeg_channels = ['C3']  # Adjust based on your data
# eog_channels = []  # Adjust based on your data

# # Perform inference
# out_infs, times, probs = eeg_infer.mne_infer(
#     inst=raw,
#     eeg=eeg_channels,
#     eog=eog_channels,
#     eeg_drop=True,
#     eog_drop=True,
#     filter=False # put back after we make a filter for real time. I just took this out to stay consistent with real time inference
# )



# log the sleep stages expected


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
    expected_stages = sleep_stages[start_epoch:2]

# compare the inferred sleep stages with the expected sleep stages


# inferred_stages = out_infs

# print the inferred stages
# print(inferred_stages)



    # print(f"Cohen's Kappa: {kappa:.4f}")




# all_predicted_classes = realtime_inference(fif_file_path)
# get combinations of eeg and eog channels
eeg_eog_combinations = make_eeg_eog_combinations(eeg_indices, eog_indices)
# make hiddens
hiddens = make_hiddens(eeg_eog_combinations)
# get eeg tensor epoched
eeg_tensor_epoched = preprocess_eeg_epoch_for_gssc(raw_csv_as_fif)
# get results for each combo
infer = make_infer()
loudest_vote, predicted_classes, class_probs, new_hiddens = get_results_for_each_combo(eeg_tensor_epoched, eeg_eog_combinations, hiddens, infer)


# print("mne-eeglab---------------")
# compare_sleep_stages(inferred_stages, expected_stages, verbose=False)
# print("old---------------")
# compare_sleep_stages(all_predicted_classes, expected_stages, verbose=False)
print("array_inference---------------")
compare_sleep_stages([loudest_vote], expected_stages, verbose=False)








