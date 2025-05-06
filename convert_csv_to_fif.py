"""
This script converts BrainFlow CSV data from a Cyton+Daisy board to MNE-Python's FIF format.
It reads the CSV file, creates an MNE Raw object with proper channel configuration,
and saves it as a FIF file. We don't need to set a montage because the channels are already 
in the correct positions and we do not need spatial information for our application. GSSC just needs channel names.

Usage:
    python convert_csv_to_fif.py input.csv output.fif
"""

import numpy as np
import pandas as pd
import mne
from brainflow.board_shim import BoardShim, BoardIds
import sys
from pathlib import Path
from gssc_local.montage import Montage


def convert_csv_to_raw(input_file):
    """
    Convert BrainFlow CSV data to FIF format
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output FIF file
    """
    # Read CSV file
    data = pd.read_csv(input_file, sep='\t', header=None)
    
    # Get board configuration
    board_id = BoardIds.CYTON_DAISY_BOARD
    sampling_rate = BoardShim.get_sampling_rate(board_id)  # 125 Hz
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    
    # Extract EEG data (channels 1-16)
    eeg_data = data.iloc[:, eeg_channels].values.T  # Convert to (n_channels, n_samples)
    
    # Convert ADC counts to microvolts for Cyton+Daisy board
    # Formula: V = (ADC_value - 8192) * (4.5 / 24) / 24
    # Where:
    # - 8192 is the zero point
    # - 4.5 is the reference voltage
    # - 24 is the gain
    # - 24 is the number of bits
    eeg_data = (eeg_data - 8192) * (4.5 / 24) / 24  # Convert to microvolts

    montage = Montage.default_sleep_montage()
    
    # Create channel names
    channel_names = montage.get_channel_labels()
    # Create channel types
    channel_types = [
        t.lower() if t.lower() in ['eeg', 'eog', 'emg', 'misc'] else 'misc'
        for t in montage.get_channel_types()
    ]
    # Create info object
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sampling_rate,
        ch_types=channel_types
    )
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info)
    
    return raw

def save_raw_to_fif(raw, output_file):
    raw.save(output_file, overwrite=True)
    print(f"Saved FIF file to: {output_file}")

def convert_csv_to_fif(input_file, output_file):
    raw = convert_csv_to_raw(input_file)
    save_raw_to_fif(raw, output_file)
    return raw

def main():
    # 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.csv', 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.fif'
    input_file = 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.csv'
    output_file = 'data/realtime_inference_test/BrainFlow-RAW_2025-03-29_23-14-54_0.fif'            
    # if len(sys.argv) != 3:
    #     print("Usage: python convert_csv_to_fif.py input.csv output.fif")
    #     sys.exit(1)
    
    # input_file = Path(input_file)
    # output_file = Path(output_file)
    
    # check if the input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    convert_csv_to_fif(input_file, output_file)
    print("Conversion complete")

if __name__ == "__main__":
    # use the file BrainFlow-RAW_2025-03-29_23-14-54_0.csv
    main() 