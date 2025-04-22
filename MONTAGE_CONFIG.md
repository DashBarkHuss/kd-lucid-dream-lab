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

| Channel | Label | Description | Filter Range (Hz) |
| ------- | ----- | ----------- | ----------------- |
| 9       | N1P   | T-EOG       | 0.1-100           |
| 10      | N2P   | B-EOG       | 0.1-100           |
| 11      | N3P   | R-LEOG      | 0.1-100           |
| 12      | N4P   | L-LEOG      | 0.1-100           |

### EMG and Other Channels

| Channel | Label | Description | Filter Range (Hz) |
| ------- | ----- | ----------- | ----------------- |
| 13      | N5P   | EMG1        | 10-100            |
| 14      | N6P   | EMG2        | 10-100            |
| 15      | N7P   | Airflow     | 0.1-50            |
| 16      | N8P   | Snoring     | 10-100            |

## Reference Configuration

- SRB (Reference): Forehead
- Ground: Mastoid
- Bias: Ground

## Signal Processing

- Notch Filters: 50Hz and 60Hz
- All channels have individual bandpass filters as specified above
- Sampling Rate: 125 Hz (Cyton) / 125 Hz (Daisy)

## Notes

- This configuration follows standard polysomnography setup
- Channels 1-8 are on the Cyton board
- Channels 9-16 are on the Daisy board
- EMG channels have a higher low-cut filter to reduce movement artifacts
- Airflow has a lower high-cut filter due to slower signal characteristics
- EOG channels maintain full bandwidth for accurate eye movement detection
- SRB (Reference) electrode placement on the forehead provides a stable reference point for all channels
