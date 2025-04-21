# Cyton/Daisy Data Format Comparison

This document compares the data formats between BrainFlow CSV (from Cyton/Daisy board) and BDF (Bio-semi Data Format) files.

## Format Comparison Table

| Feature                | BrainFlow CSV                          | BDF                                          |
| ---------------------- | -------------------------------------- | -------------------------------------------- |
| File Extension         | .csv                                   | .bdf                                         |
| Data Format            | Text-based, comma-separated            | Binary                                       |
| Channels & Sampling    | - Ganglion: 4 channels @ 200Hz         | - Ganglion: 4 channels @ 200Hz               |
| Rate by Board          | - Cyton: 8 channels @ 250Hz            | - Cyton: 8 channels @ 250Hz                  |
|                        | - Cyton+Daisy: 16 channels @ 125Hz     | - Cyton+Daisy: 16 channels @ 125Hz           |
| Timestamp Format       | Unix timestamp (seconds)\*             | Relative time from recording start (seconds) |
| Annotations/Events     | Separate marker column                 | Embedded annotations track                   |
| File Size              | Larger (text-based)                    | Smaller (binary)                             |
| Software Compatibility | BrainFlow, Python, general CSV readers | MNE, EDF/BDF viewers                         |
| Metadata Storage       | Limited (in CSV headers)               | Extensive header information                 |

## Converting Between Formats

To convert between these formats, you can use tools like:

- MNE-Python
- BrainFlow's data conversion utilities

## Notes

- \*BrainFlow CSV timestamps: During playback, BrainFlow generates new timestamps rather than using the original recording timestamps
- BDF timestamps are relative to recording start, incrementing based on the sampling rate
- BDF files contain more extensive metadata and header information
- CSV format is more accessible for basic data analysis but less space-efficient
- BDF is part of the EDF+ family of formats commonly used in sleep research

For more detailed conversion methods, see our conversion scripts:

- `make_fif_out_of_set.py`
- `make_fif_file.py`

An example of how to convert bdf to csv is in the `convert_bdf_time_to_brainflow_csv.py` script.

" These packets are numbered consecutively and can help the GUI receiving code tell if any packets are dropped."
It sounds like the packets are only used for alerting the user of packet loss. So do the packets matter for brainflow streaming or playpack? Could I just ignore them if I'm converting a bdf to a brainflow csv?

## BDF Format

You can see the confirmed format of the BDF file here: https://www.biosemi.com/faq/file_format.htm

If the linked file contradicts anything in this file, the linked file is correct.
