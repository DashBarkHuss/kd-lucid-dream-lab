# print the data of each brainflow file in the folder. only works for cyton-daisy files. Need to change timestamp column for other boards.
import pandas as pd
import numpy as np
import os

# Config
data_folder_path = '/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/realtime_inference_test/OpenBCISession_2025-03-29_23-14-54'
files = sorted(os.listdir(data_folder_path))  # sort for consistency
# filter out files that don't end with .csv
files = [file for file in files if file.endswith('.csv')]
timestamp_column = 30
sample_rate = 125
sample_rate_seconds = 1 / sample_rate

# Time formatting
def convert_to_hawaiian_time(unix_timestamp):
    if pd.isna(unix_timestamp):
        return "NaN HST"
    hawaiian_offset = 10 * 3600
    hawaiian_time = unix_timestamp - hawaiian_offset
    total_seconds = int(hawaiian_time)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    period = "AM" if hours < 12 else "PM"
    hours = 12 if hours % 12 == 0 else hours % 12
    return f"{hours:02d}:{minutes:02d}:{seconds:02d} {period} HST"

# Read first file for comparison
first_file_data = pd.read_csv(os.path.join(data_folder_path, files[0]), sep='\s+', header=None)

# Process each file
for file in files:
    print(f"\n=== Processing File: {file} ===")
    try:
        file_path = os.path.join(data_folder_path, file)
        eeg_data = pd.read_csv(file_path, sep='\s+', header=None)
        print(f"Rows: {len(eeg_data)}")
        # Print overall start and end time in Hawaiian
        start_unix = eeg_data.iloc[0, timestamp_column]
        end_unix = eeg_data.iloc[-1, timestamp_column]
        print(f"Start time (Hawaiian): {convert_to_hawaiian_time(start_unix)}")
        print(f"End time (Hawaiian): {convert_to_hawaiian_time(end_unix)}")

        # ---- GAP DETECTION ----
        time_diffs = eeg_data.iloc[:, timestamp_column].diff().fillna(0)
        gaps = time_diffs[(time_diffs < sample_rate_seconds - 3) | (time_diffs > sample_rate_seconds + 3)]
        gaps_list = [{"index": i, "gap_seconds": gaps[i]} for i in gaps.index if gaps[i] != 0]
        gaps_df = pd.DataFrame(gaps_list)

        # Annotate gaps
        if not gaps_df.empty:
            gaps_df['start_time_seconds'] = [eeg_data.iloc[i - 1, timestamp_column] for i in gaps_df['index']]
            gaps_df['end_time_seconds'] = [eeg_data.iloc[i, timestamp_column] for i in gaps_df['index']]
            gaps_df['start_time_hawaiian'] = gaps_df['start_time_seconds'].apply(convert_to_hawaiian_time)
            gaps_df['end_time_hawaiian'] = gaps_df['end_time_seconds'].apply(convert_to_hawaiian_time)
            print("\nGaps found (with Hawaiian time):")
            print(gaps_df[['index', 'gap_seconds', 'start_time_hawaiian', 'end_time_hawaiian']])
        else:
            print("No gaps found.")

        # ---- SEGMENT DETECTION ----
        segment_bounds = [0] + [gap['index'] for gap in gaps_list] + [len(eeg_data)]
        segments = [(segment_bounds[i], segment_bounds[i + 1]) for i in range(len(segment_bounds) - 1)]
        segments = [s for s in segments if s[1] > s[0]]

        print("\nContinuous data segments (Hawaiian time):")
        for i, (start_idx, end_idx) in enumerate(segments):
            start_time = eeg_data.iloc[start_idx, timestamp_column]
            end_time = eeg_data.iloc[end_idx - 1, timestamp_column]
            print(f"Segment {i+1}: {convert_to_hawaiian_time(start_time)} → {convert_to_hawaiian_time(end_time)}")

        print(f"\nTotal gaps: {len(gaps_df)}")
        print(f"Total segments: {len(segments)}")  # should be gaps + 1

        # ---- CHECK ROW OVERLAP WITH FIRST FILE ----
        if file != files[0]:
            number_of_rows = 10000
            last_rows = first_file_data.iloc[-number_of_rows:]
            rows1 = last_rows.reset_index(drop=True).to_numpy()
            rows2 = eeg_data.iloc[-number_of_rows:].reset_index(drop=True).to_numpy()

            if np.allclose(rows1, rows2):  # Adjust tolerance if needed
                print(f"The last {number_of_rows} rows of {file} are all THE SAME as the last {number_of_rows} rows of the first file")
            else:
                print(f"The last {number_of_rows} rows of {file} are NOT all the same as the last {number_of_rows} rows of the first file")
                diff = pd.DataFrame(rows1 - rows2)
                print("Difference between last rows of first file and current file:")
                print(diff)


    except Exception as e:
        print(f"Error processing file {file}: {e}")