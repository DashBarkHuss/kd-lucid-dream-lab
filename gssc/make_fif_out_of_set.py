"""
This script converts an EEGLAB .set file to MNE-Python's .fif format.
It reads the specified .set file, loads it into memory with EOG channels identified,
saves it as a .fif file, and then reads the newly created .fif file to verify.
This conversion allows for easier processing of EEG data using MNE-Python tools.
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

# Path to save the .fif file
fif_file_path = 'data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_full_raw.fif'

# Save the sliced data as a .fif file
raw.save(fif_file_path, overwrite=True)

# read the fif file
raw = mne.io.read_raw_fif(fif_file_path)
         
