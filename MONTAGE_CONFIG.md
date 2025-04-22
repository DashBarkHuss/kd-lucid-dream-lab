# OpenBCI Cyton+Daisy Sleep Montage Configuration

## Channel Configuration

### EEG Channels (0.1-100 Hz Bandpass)

| Channel | Label | Location        | Filter Range (Hz) |
| ------- | ----- | --------------- | ----------------- |
| 1       | F3    | Frontal Left    | 0.1-100           |
| 2       | F4    | Frontal Right   | 0.1-100           |
| 3       | C3    | Central Left    | 0.1-100           |
| 4       | C4    | Central Right   | 0.1-100           |
| 5       | O1    | Occipital Left  | 0.1-100           |
| 6       | O2    | Occipital Right | 0.1-100           |
| 7       | T5    | Temporal Left   | 0.1-100           |
| 8       | T6    | Temporal Right  | 0.1-100           |

### EOG Channels (0.1-100 Hz Bandpass)

| Channel    | Label  | Description           | Filter Range (Hz) |
| ---------- | ------ | --------------------- | ----------------- |
| 9 (Daisy)  | T-EOG  | Top Horizontal EOG    | 0.1-100           |
| 10 (Daisy) | B-EOG  | Bottom Horizontal EOG | 0.1-100           |
| 11 (Daisy) | R-LEOG | Right Lateral EOG     | 0.1-100           |
| 12 (Daisy) | L-LEOG | Left Lateral EOG      | 0.1-100           |

### EMG and Other Channels

| Channel | Label | Description | Filter Range (Hz) |
| 13 (Daisy) | EMG1 | Center of Chin | 10-100 |
| 14 (Daisy) | EMG2 | To the side of the chin | 10-100 |
| 15 (Daisy) | Airflow | Airflow | 0.1-50 |
| 16 (Daisy) | Snoring | Snoring | 10-100 |

## Reference Configuration

- Reference: Forehead. Pin: Bottom SRM
- Ground: Mastoid. Pin: Bottom Bias

## Signal Processing

- Notch Filters: 50Hz and 60Hz
- All channels have individual bandpass filters as specified above
- Sampling Rate: 125 Hz (Cyton_DAISY)

## Notes

- This configuration follows standard polysomnography setup
- Channels 1-8 are on the Cyton board
- Channels 9-16 are on the Daisy board
- EMG channels have a higher low-cut filter to reduce movement artifacts
- Airflow has a lower high-cut filter due to slower signal characteristics
- EOG channels maintain full bandwidth for accurate eye movement detection
- SRB (Reference) electrode placement on the forehead provides a stable reference point for all channels
