"""
This script compares sleep stage classification results from different inference methods.
It processes EEG data from a .set file, converts it to .fif format, and runs both MNE-based
and array-based GSSC inference algorithms. The script then compares the predicted sleep stages
with ground truth data from a .mat file, calculating accuracy metrics and providing detailed
analysis of classification performance across different sleep stages.

This file uses gssc_array_inference_with_fif.py to get the predicted classes.
"""

import mne
from gssc.infer import EEGInfer
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# import realtime inference from gssc_array_infer.py in this directory
from gssc_helper import realtime_inference


# Path to your .set file
set_file_path = '/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_corrected_to_1000hz.set'

# Read the .set file
raw = mne.io.read_raw_eeglab(
    input_fname=set_file_path,
    eog=['L-HEOG', 'R-HEOG', 'B-VEOG'],  # Specify EOG channels if needed
    preload=True,  # Load data into memory
    verbose=True  # Enable verbose output for debugging
)


start = 3000
end = 6000

# Slice the data 
raw_sliced = raw.copy().crop(tmin=start, tmax=end) 

eeg_channels = ['F3','C3', 'O1']  #  Adjust based on your data
eog_channels = ['L-HEOG']  # Adjust based on your data
# eeg_channels = ['C3']  # Adjust based on your data
# eog_channels = ['L-HEOG', 'R-HEOG']  # Adjust based on your data


# Path to save the .fif file
fif_file_path = '/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_raw.fif'

# Save the sliced data as a .fif file
raw_sliced.save(fif_file_path, overwrite=True)

# Load the .fif file
raw = mne.io.read_raw_fif(fif_file_path, preload=True)

# Then read it to get the correct channel names
raw = mne.io.read_raw_fif(fif_file_path, preload=True)
channel_names = raw.ch_names

# Create indices from the newly saved file
eeg_indices = [i for i, ch in enumerate(channel_names) if ch in eeg_channels]
eog_indices = [i for i, ch in enumerate(channel_names) if ch in eog_channels]

# Initialize the EEGInfer class
eeg_infer = EEGInfer()

# Specify the EEG and EOG channels
# I'm testing with only C3 to get ArrayInfer to match, usually you should use the comment out portion below
# eeg_channels = ['C3']  # Adjust based on your data
# eog_channels = []  # Adjust based on your data

# Perform inference
out_infs, times, probs = eeg_infer.mne_infer(
    inst=raw,
    eeg=eeg_channels,
    eog=eog_channels,
    eeg_drop=True,
    eog_drop=True,
    filter=False # put back after we make a filter for real time. I just took this out to stay consistent with real time inference
)



# log the sleep stages expected


# Path to your .mat file
mat_file_path = '/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/sleep_data/dash_data_104_session_6/alpha104ses06scoringKK.mat'

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


inferred_stages = out_infs

# print the inferred stages
# print(inferred_stages)

def compare_sleep_stages(inferred_stages, expected_stages, verbose=True):
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
    # TODO: removed unscored data or maye also movement/undefined epochs from comparison
    # Ensure both arrays are the same length
    if len(inferred_stages) != len(expected_stages):
        raise ValueError("Inferred and expected stage arrays must be the same length")
    
    # remove the epochs where the expected stage is 7 or 6
    # First get the epoch index of all expected stages that are 7 or 6
    # then remove those epochs from the inferred stages 
    remove_epochs = np.where((expected_stages == 7) | (expected_stages == 6))[0]
    inferred_stages_cut = np.delete(inferred_stages, remove_epochs)
    expected_stages_cut = np.delete(expected_stages, remove_epochs)

    # Initialize counters
    matches = 0
    total = len(inferred_stages_cut)
    mismatches = []

    # Compare stages epoch by epoch
    print("")
    max_length = max(len(f"Epoch {len(inferred_stages)}: ") + 20, 30)  # Adjust 20 if needed

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
    
    # kappa = cohen_kappa_score(expected_stages, inferred_stages)

    # Print summary
    print(f"\nTotal Epochs: {total}")
    print(f"Matched Epochs: {matches}")
    print(f"Mismatched Epochs: {total - matches}")
    print(f"Accuracy: {(accuracy*100):.4f}%")

    if verbose:
        false_pos_rem_count = [mismatch for mismatch in mismatches if (mismatch[1] == 4.0)]
        false_neg_rem_count = [mismatch for mismatch in mismatches if (mismatch[2] == 5.0)]

        false_pos_rem_prc = len(false_pos_rem_count) / total
        false_neg_rem_prc = len(false_neg_rem_count) / total
        overall_rem_accuracy = 1-((len(false_pos_rem_count) + len(false_neg_rem_count)) / total)

        false_pos_n1_count = [mismatch for mismatch in mismatches if (mismatch[1] == 1.0)]
        false_neg_n1_count = [mismatch for mismatch in mismatches if (mismatch[2] == 1.0)]

        false_pos_n1_prc = len(false_pos_n1_count) / total
        false_neg_n1_prc = len(false_neg_n1_count) / total
        overall_n1_accuracy = 1-((len(false_pos_n1_count) + len(false_neg_n1_count)) / total)

        false_pos_n2_count = [mismatch for mismatch in mismatches if (mismatch[1] == 2.0)]
        false_neg_n2_count = [mismatch for mismatch in mismatches if (mismatch[2] == 2.0)]

        false_pos_n2_prc = len(false_pos_n2_count) / total
        false_neg_n2_prc = len(false_neg_n2_count) / total
        overall_n2_accuracy = 1-((len(false_pos_n2_count) + len(false_neg_n2_count)) / total)

        false_pos_n3_count = [mismatch for mismatch in mismatches if (mismatch[1] == 3.0)]
        false_neg_n3_count = [mismatch for mismatch in mismatches if (mismatch[2] == 3.0)]

        false_pos_n3_prc = len(false_pos_n3_count) / total
        false_neg_n3_prc = len(false_neg_n3_count) / total
        overall_n3_accuracy = 1-((len(false_pos_n3_count) + len(false_neg_n3_count)) / total)

        false_pos_wake_count = [mismatch for mismatch in mismatches if (mismatch[1] == 0.0)]
        false_neg_wake_count = [mismatch for mismatch in mismatches if (mismatch[2] == 0.0)]

        false_pos_wake_prc = len(false_pos_wake_count) / total
        false_neg_wake_prc = len(false_neg_wake_count) / total
        overall_wake_accuracy = 1-((len(false_pos_wake_count) + len(false_neg_wake_count)) / total)
        print("")
        print(f"False Positive REM Epochs: {len(false_pos_rem_count)}")
        print(f"False Positive REM percentage: {(false_pos_rem_prc * 100):.2f}%")
        print(f"False Negative REM Epochs: {len(false_neg_rem_count)}")
        print(f"False Negative REM:  {(false_neg_rem_prc * 100):.2f}%")
        print(f"Overall REM Accuracy: {(overall_rem_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N1 Epochs: {len(false_pos_n1_count)}")
        print(f"False Positive N1 percentage: {(false_pos_n1_prc * 100):.2f}%")
        print(f"False Negative N1 Epochs: {len(false_neg_n1_count)}")
        print(f"False Negative N1 percentage: {(false_neg_n1_prc * 100):.2f}%")
        print(f"Overall N1 Accuracy: {(overall_n1_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N2 Epochs: {len(false_pos_n2_count)}")
        print(f"False Positive N2 percentage: {(false_pos_n2_prc * 100):.2f}%")
        print(f"False Negative N2 Epochs: {len(false_neg_n2_count)}")
        print(f"False Negative N2 percentage: {(false_neg_n2_prc * 100):.2f}%")
        print(f"Overall N2 Accuracy: {(overall_n2_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N3 Epochs: {len(false_pos_n3_count)}")
        print(f"False Positive N3 percentage: {(false_pos_n3_prc * 100):.2f}%")
        print(f"False Negative N3 Epochs: {len(false_neg_n3_count)}")
        print(f"False Negative N3 percentage: {(false_neg_n3_prc * 100):.2f}%")
        print(f"Overall N3 Accuracy: {(overall_n3_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive Wake Epochs: {len(false_pos_wake_count)}")
        print(f"False Positive Wake percentage: {(false_pos_wake_prc * 100):.2f}%")
        print(f"False Negative Wake Epochs: {len(false_neg_wake_count)}")
        print(f"False Negative Wake percentage: {(false_neg_wake_prc * 100):.2f}%")
        print(f"Overall Wake Accuracy: {(overall_wake_accuracy * 100):.2f}%")
        print("")
    print(f"Removed Unscored and Movement/Undefined Epochs: {len(remove_epochs)}")


    # print(f"Cohen's Kappa: {kappa:.4f}")




# all_predicted_classes = realtime_inference(fif_file_path)
loudest_votes, predicted_classes_list, class_probs_list = realtime_inference(raw_sliced, eeg_channels, eog_channels)
print("mne-eeglab---------------")
compare_sleep_stages(inferred_stages, expected_stages, verbose=False)
# print("old---------------")
# compare_sleep_stages(all_predicted_classes, expected_stages, verbose=False)
print("array_inference---------------")
compare_sleep_stages(loudest_votes, expected_stages, verbose=False)

