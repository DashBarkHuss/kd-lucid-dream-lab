"""
This script converts a segment of an EEGLAB .set file to MNE-Python's .fif format.
It reads the specified .set file, extracts a 30-second segment (from 4200s to 4230s),
saves it as a .fif file, and verifies the conversion. The script also extracts and 
displays corresponding sleep stage information from a .mat file for the selected time segment.
"""

import mne

# Path to your .set file
set_file_path = 'data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_corrected_to_1000hz.set'

# Read the .set file
raw = mne.io.read_raw_eeglab(
    input_fname=set_file_path,
    eog=['L-HEOG', 'R-HEOG', 'B-VEOG'],  # Specify EOG channels if needed
    preload=True,  # Load data into memory
    verbose=True  # Enable verbose output for debugging
)


start = 4200
end = 4230

# Slice the data 
raw_sliced = raw.copy().crop(tmin=start, tmax=end) 


# Path to save the .fif file
fif_file_path = 'data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_raw.fif'

# Save the sliced data as a .fif file
raw_sliced.save(fif_file_path, overwrite=True)

# read the fif file
raw_sliced = mne.io.read_raw_fif(fif_file_path)
         
# log the sleep stages for the first 60 seconds which can be found in the .mat in stageData/stages
import h5py
import numpy as np

# Path to your .mat file
mat_file_path = 'data/sleep_data/dash_data_104_session_6/alpha104ses06scoringKK.mat'

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

# Convert time window to epoch indices
epoch_duration = 30  # Each epoch is 30 seconds
start_epoch = int(start // epoch_duration)
end_epoch = int(end // epoch_duration)

# Select stages for the specified time segment
stages_for_segment = sleep_stages[start_epoch:end_epoch]

# Log the sleep stages
print("Sleep stages for segment:")
for i, stage in enumerate(stages_for_segment):
    print(f"Epoch {i+1}: Stage {stage}")


# log channel names
print("Channel names:", raw_sliced.info['ch_names'])


